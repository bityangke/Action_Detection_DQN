-- do not mask when extract C3D
-- do not split ground truth
-- no dist 
-- jump
-- no expert
-- for thomas
-- no narrow when 16

require 'Hjj_Read_Input_Cmd'
require 'Hjj_Reinforcement'
require 'Zt_Interface_new'
require 'Hjj_Mask_and_Actions'
require 'Hjj_Metrics'
require 'optim'

--read input para
local cmd = torch.CmdLine()
opt = func_read_training_cmd(cmd, arg)

-- create log file
local log_file = io.open(opt.log_log, 'w')
if not log_file then
	print("open log file error")
	error("open log file error")
end

-- read training clip id from files
local training_file = './' .. opt.data_path .. '/trainlist_id.t7'
print(training_file)
local clip_table = torch.load(training_file)
local tt = clip_table[opt.class]
if tt == nil then
	error('no trainlist file')
end

-- thomas
local training_clip_table={}
for i=1,10 do
	table.insert(training_clip_table, tt[#tt-10+i])
end

--set training parameters
local max_epochs = opt.epochs
local batch_size = opt.batch_size
local max_steps = 15

--DQN training trick parameters
local experience_replay_buffer_size = opt.replay_buffer
local gamma = 0.90 --discount factor
local epsilon = 1 -- greedy policy
local trigger_thd = 0.5 -- threshold for terminal

local count_train = torch.Tensor(1):fill(0)
local train_period = torch.floor(opt.batch_size/100)

-- if a object was masked more than 0.6, than not used anymore
-- it is overlap not iou; bad operation may lead to overlap more than
-- mask_thd while iou is less than trigger_thd 
local mask_thd = 0.6 
-- number_of_actions and history_action_buffer_size are globle variables in Hjj_Reinforcement
local history_vector_size = number_of_actions * history_action_buffer_size
local input_vector_size = history_vector_size + C3D_size
-- define the last action as trigger
local trigger_action = number_of_actions
local jump_action = number_of_actions-1
local act_alpha = opt.alpha

--init replay memory
local replay_memory = {}

--init reward
local reward = 0
--if opt.model_name is '0', then init DQN model
--else load a saved DQN
local dqn = {}
dqn, opt.gpu = func_get_dqn(opt.model_name, opt.gpu, log_file)
local params, gradParams = dqn:getParameters()

-- set gpu
opt.gpu = func_set_gpu(opt.gpu, log_file)

if opt.gpu >=0 then dqn = dqn:cuda() end
-- define loss function and trainer
local criterion = nn.MSECriterion()
if opt.gpu >=0 then crirerion = criterion:cuda() end

-- training with optim
-- optim paras for optim
local optimState = {learningRate = opt.lr, maxIteration = 1, learningRateDecay = 0.00005, evalCounter = 0}
--local optimState = {learningRate = opt.lr, maxIteration = 1}
local logger = optim.Logger(opt.log_err)
logger:setNames{'Training_error', 'epoch'}


--read dataset
local gt_table = func_get_data_set_info(opt.data_path, opt.class, 1)
--print(gt_table)
local max_gt_length = 128 -- max length to split gt
--gt_table = func_modify_gt(gt_table, max_gt_length)

-- load C3D model
local C3D_m = torch.load('c3d.t7');
for i = 1, max_epochs
do
	log_file:write('It is the ' .. i .. ' epoch\n')
	print('It is the ' .. i .. ' epoch')
	
	for j, v in pairs(training_clip_table)
	do
		
		local masked = false 
		local masked_segs={}
		local not_finished = true
		local tmp_gt = gt_table[v]
		local total_frms = tmp_gt[1][3]
		local gt_num = table.getn(tmp_gt)
		local available_objects = torch.Tensor(gt_num):fill(1)

		
		log_file:write('\tIt is the ' .. j .. ' clip, clip_id = ' .. 
						v .. ' total_frms = '.. total_frms .. '\n')
		print('\tIt is the '.. j .. ' clip, clip_id = ' .. 
						v .. ' total_frms = '.. total_frms)
		
		for k = 1, 40
		do
			log_file:write('\t\tIt is the ' .. k .. ' gt, from '.. '\n')
			--				tmp_gt[k][1] ..' to '.. tmp_gt[k][2] .. '\n')
			print('\t\tIt is the ' .. k .. ' gt, from '.. '\n')
			--				tmp_gt[k][1] ..' to '.. tmp_gt[k][2])
			
			-- init mask, return beg index and end index of mask
			local cur_mask = func_mask_random_init(total_frms, masked_segs)
			local old_mask = cur_mask
			
			-- iou_table record the iou of each gt and cur_mask
			-- reset iou_table in the beginning of each loop
			local iou_table = torch.Tensor(gt_num):fill(0)
			--local dist_table = torch.Tensor(gt_num):fill(max_dist)
			local old_iou = 0
			local new_iou = 0
			local overlap = 0
			--local old_dist = max_dist -- max_dist is a global variance in Metric
			local new_dist = max_dist
			
			
			if masked then
			-- an object has been found in last loop
				for p=1, gt_num
				do
					local tmp_iou = func_calculate_overlapping(old_mask, tmp_gt[p])
					if tmp_iou > mask_thd then
						available_objects[p] = 0
					end
				end -- for
			end -- if
			
			-- check if available objects left
			if torch.nonzero(available_objects):numel() == 0 then
				not_finished = false
			end
			
			-- calculate iou for cur_mask and gt
			old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
			overlap = func_calculate_overlapping(tmp_gt[index], cur_mask) -- intersec/cur_mask
			--old_dist, new_dist, dist_table,old_iou, new_iou, iou_table, index  = 
			--			func_follow_dist_iou(cur_mask, tmp_gt, available_objects,iou_table,dist_table)
			

			local now_target_gt = tmp_gt[index]
			
			-- init history action buffer
			local history_vector = torch.Tensor(history_vector_size):fill(0)
			-- get C3D
			local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
											 v, cur_mask[1], cur_mask[2], C3D_m, {})
			local input_vector = torch.cat(C3D_vector, history_vector, 1)
			
			if opt.gpu >=0 then input_vector = input_vector:cuda() end
			
			local bingo = false -- it is a right trigger action or not
			local action = 0 -- init action
			local step_count = 0 -- reset step_count
			reward = 0 -- re-init reward
			while (not bingo) and (step_count < max_steps) and not_finished
			do
				log_file:write('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							 .. overlap .. '\n')
				print('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlapt = '
							  .. overlap .. '\n')
				-- run DQN
				
				local action_output = dqn:forward(input_vector)
				print(action_output)
				local tmp_flag = 0
				
				-- It is checking for last non-trigger action, which may actually lead to an 
				-- terminal state; we force it to be terminal action in case actual IoU 
				-- is higher than 0.5, to train faster the agent; 
				local tmp_v = 0
				tmp_v, action = torch.max(action_output,1)
				action = action[1]-- from tensor to numeric type
				if (cur_mask[2]-cur_mask[1]+1) >= max_gt_length*2 and action == 4 then
					-- forbid expand than max_gt_length
					-- choose a random action
					action = torch.random(torch.Generator(),1,3)
				elseif (cur_mask[2]-cur_mask[1]) <= 16 and action == 3 then
					action = torch.random(torch.Generator(),1,3)
					if action == 3 then action = 4 end
				end
				if action == trigger_action then 
					tmp_flag = 1 
				elseif i < max_epochs and new_iou > trigger_thd then
					action = trigger_action
				--elseif overlap > trigger_thd2 and (cur_mask[2]-cur_mask[1]+1) > trigger_len and i < max_epochs then
				--	action = trigger_action
				elseif i < max_epochs and	 new_iou == 0 then
					action = jump_action
				elseif torch.uniform(torch.Generator()) < epsilon then -- greedy policy
					action = torch.random(torch.Generator(),1,number_of_actions)
				end
				
				if action == trigger_action then -- estemated as trigger
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)
					--old_dist, new_dist, dist_table,old_iou, new_iou, iou_table, index  = 
					--	func_follow_dist_iou(cur_mask, tmp_gt, available_objects,iou_table,dist_table)
					
					now_target_gt = tmp_gt[index]
					--if overlap > trigger_thd2 and (cur_mask[2]-cur_mask[1]+1) > trigger_len then
						-- if satisfy overlapping condition, give positive reward
					--	reward = func_get_reward_trigger(1)
					--else
						-- give reward according to iou
						reward = func_get_reward_trigger(new_iou)
					--end
					step_count = step_count+1
					bingo = true
					
					log_file:write('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							 .. overlap .. '; self = '.. tmp_flag .. '\n')
					print('\t\t\tStep: ' .. step_count .. ' ---> Action= ' .. action ..
							' ; Mask= [' .. cur_mask[1] .. ' , ' .. cur_mask[2] .. 
							' ]; GT = [' .. now_target_gt[1] .. ' , ' .. now_target_gt[2] .. 
							 ' ]; Reward= ' .. reward .. ' ; iou = ' .. new_iou .. '; overlap = '
							  .. overlap .. '; self = '.. tmp_flag ..'\n')
				elseif action == jump_action then
					-- encourage jump action if it is a iou==0 state
					if new_iou == 0 then
						reward = func_get_reward_movement(0, 1,0,0) -- half reward
					else
						reward = func_get_reward_movement(1,0,0,0)
					end
					cur_mask = func_take_action(cur_mask, action, total_frms, act_alpha)
					--old_dist, new_dist, dist_table,old_iou, new_iou, iou_table, index  = 
					--	func_follow_dist_iou(cur_mask, tmp_gt, available_objects,iou_table,dist_table)
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)
					now_target_gt = tmp_gt[index]
					old_iou = new_iou
					--old_dist = new_dist
					history_vector = func_update_history_vector(history_vector, action)
					step_count = step_count + 1
				else -- take action
				-- 1 move forward; 2 move back; 4 expand; 3 narrow
					cur_mask = func_take_action(cur_mask, action, total_frms, act_alpha)
					--old_dist, new_dist, dist_table,old_iou, new_iou, iou_table, index  = 
					--	func_follow_dist_iou(cur_mask, tmp_gt, available_objects,iou_table,dist_table)
					old_iou, new_iou, iou_table, index = func_follow_iou(cur_mask,
												tmp_gt, available_objects, iou_table)
					overlap = func_calculate_overlapping(tmp_gt[index], cur_mask)
					now_target_gt = tmp_gt[index]
					reward = func_get_reward_movement(old_iou, new_iou,0,0)
					old_iou = new_iou
					--old_dist = new_dist
					history_vector = func_update_history_vector(history_vector, action)
					step_count = step_count + 1
					-- log wiil be written at the beginning of the next loop
				end
				C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
											 v, cur_mask[1], cur_mask[2],C3D_m, {})
				local new_input_vector = torch.cat(C3D_vector, history_vector, 1)
				if opt.gpu >=0 then new_input_vector = new_input_vector:cuda() end
				
				count_train[1] = count_train[1]+1
				-- experience replay
				local tmp_experience  = {input_vector, action, reward, new_input_vector}
				if table.getn(replay_memory) < experience_replay_buffer_size then
					table.insert(replay_memory, tmp_experience)
					input_vector = new_input_vector
				else
					-- replay_memory is a stack
					table.remove(replay_memory, 1)
					table.insert(replay_memory, tmp_experience)
					
					local tmp_mod = torch.fmod(count_train,train_period)
					tmp_mod = tmp_mod[1]
					if tmp_mod == 0 then
						--local minibatch = func_sample(replay_memory, batch_size-3) -- in Hjj_Reinforcement
						--local expert_batch = func_sample(expert_experience, 3)
						--for l = 1,3 do table.insert(minibatch,expert_batch[l]) end
						local minibatch = func_sample(replay_memory, batch_size) 
						local memory = {}
						-- construct training set
						local training_set = {data=torch.Tensor(batch_size, input_vector_size),
												 label=torch.Tensor(batch_size, number_of_actions)}
						function training_set:size() return batch_size end
						setmetatable(training_set, {__index = function(t,i) 
															return {t.data[i], t.label[i]} end})
					
						if opt.gpu >= 0 then 
							training_set.data = training_set.data:cuda()
							training_set.label = training_set.label:cuda() 
						end
					
						log_file:write('\t\t\t\t Doing memory replay...\n')
						print('\t\t\t\t Doing memory replay...\n')
						for l, memory in pairs(minibatch)
						do
							local tmp_input_vector = memory[1]
							local tmp_action = memory[2]
							local tmp_reward = memory[3]
							local tmp_new_input_vector = memory[4]
							local old_action_output = dqn:forward(tmp_input_vector)
							local new_action_output = dqn:forward(tmp_new_input_vector)
							local tmp_v = 0
							local tmp_index = 0
							local y = old_action_output:clone()
							tmp_v, tmp_index = torch.max(new_action_output, 1)
							tmp_v = tmp_v[1]
							tmp_index = tmp_index[1]
							local update_reward = 0
							if tmp_action == trigger_action then
								update_reward = tmp_reward
							else
								update_reward = tmp_reward + gamma * tmp_v
							end
							y[tmp_action] = update_reward
							training_set.data[l] = tmp_input_vector
							training_set.label[l] = y
						end
						-- training
						log_file:write('\t\t\t\t Training...\n')
						print('\t\t\t\t Training...\n')
						local function feval(params)
							gradParams:zero()
						
							local outputs = dqn:forward(training_set.data)
							local loss = criterion:forward(outputs, training_set.label)
							local dloss_doutputs = criterion:backward(outputs, training_set.label)
							dqn:backward(training_set.data, dloss_doutputs)
							logger:add{loss*100, i}
							return loss, gradParams
						end
						optimState.evalCounter = optimState.evalCounter + 1
						optim.sgd(feval, params, optimState)
					end -- mod
					input_vector = new_input_vector
				end -- if memory replay
				
				if action == trigger_action then
					bingo = true
					masked = true
					if reward == 3 then
						table.insert(masked_segs, {cur_mask[1]+torch.floor((cur_mask[2]-cur_mask[1]+1)*0.1),
									cur_mask[2]-torch.floor((cur_mask[2]-cur_mask[1]+1)*0.1)})
					end
				else
					masked = false
				end
			end -- while (not bingo) and (step_count < max_steps) and not_finished
			-- available_objects[index] = 0
		end -- gts loop
		-- visualize training error
		logger:style{'+-'}
		logger:plot()
	end -- clips loop
	if epsilon > 0.1 then
		epsilon = epsilon - 0.1
	end
	-- save enviroments
	if table.getn(replay_memory) >= experience_replay_buffer_size and i > 4 then
		local mdl_name={}
		if opt.gpu >= 0 then
			mdl_name = './model/g_'.. opt.name .. opt.class .. '_'.. i
		else 
			mdl_name = './model/c_'.. opt.name .. opt.class .. '_'.. i
		end
		torch.save(mdl_name, {dqn = dqn, gpu = opt.gpu})
	end

end -- epochs loop

log_file:close()




















