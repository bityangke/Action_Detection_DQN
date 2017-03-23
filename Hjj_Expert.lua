require 'torch'
require 'Zt_Interface'
require 'Hjj_Reinforcement'
require 'Hjj_Metrics'
require 'optim'
require 'Hjj_Read_Input_Cmd'

--read input para
local cmd = torch.CmdLine()
opt = func_read_expert_cmd(cmd, arg)

local log_file = io.open('./finetune_model/log', 'w')
if not log_file then
	print("open log file error")
	error("open log file error")
end


-- load dqn
if opt.model_name == '0' then
	error('model needed')
end
local dqn={}
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
local optimState = {learningRate = opt.lr, maxIteration = 1, learningRateDecay = 0.02, evalCounter = 0}
local logger = optim.Logger('./finetune_model/log_err')
logger:setNames{'Training_error', 'epoch'}

-- prepare expert memory
local batch_size = opt.batch_size
local replay_memory = {} 
local history_vector_size = number_of_actions * history_action_buffer_size
local input_vector_size = history_vector_size + C3D_size

if opt.load_memory > 0 then
	replay_memory = torch.load(opt.data_path)
	replay_memory = replay_memory.replay_memory
	print('replay memory size ' .. #replay_memory)
else
	-- load C3D model
	local C3D_m = torch.load('c3d.t7');

	-- load clip table
	local training_file = './' .. opt.data_path .. '/trainlist_id.t7'
	print(training_file)
	local clip_table = torch.load(training_file)
	local training_clip_table = clip_table[opt.class]
	if training_clip_table == nil then
		error('no trainlist file')
	end
	
	local max_gt_length = 128
	--gt_table
	local gt_table = func_get_data_set_info(opt.data_path, opt.class, 1)
	gt_table = func_modify_gt(gt_table, max_gt_length)
	
	--print(gt_table)
	--error()
	
	for i, v in pairs(training_clip_table)
	do
		print(i .. '\t' .. v)
		local tmp_gt = gt_table[v]
		for j=1,#tmp_gt
		do
			print(j)
			local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
											 v, tmp_gt[j][1], tmp_gt[j][2], C3D_m, {})
			-- create random action history								 
			local trigger_action =  number_of_actions
			local history_vector = torch.Tensor(history_vector_size):fill(0)
			for k=1,history_action_buffer_size
			do
				history_vector[torch.random(torch.Generator(),1,number_of_actions) + 
								number_of_actions * (k-1)] = 1
			end
			local input_vector = torch.cat(C3D_vector, history_vector, 1)
			if opt.gpu >=0 then input_vector = input_vector:cuda() end
			table.insert(replay_memory, {input_vector, trigger_action, 
										func_get_reward_trigger(1), input_vector})
		end
	end
	torch.save('./finetune_model/expert_experience', {replay_memory = replay_memory})
end -- if opt.load_memory


local gamma = 0.90 --discount factor
-- training
for i=1,opt.epochs
do
	print('epoch ' .. i .. '..')
	for l=1,30 do
		local minibatch = func_sample(replay_memory, batch_size) -- in Hjj_Reinforcement
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
						optimState.evalCounter = optimState.evalCounter+1
						optim.sgd(feval, params, optimState)
		end
	local mdl_name={}
	if opt.gpu >= 0 then
		mdl_name = './finetune_model/g_'.. opt.class .. '_'.. i
	else
		mdl_name = './finetune_model/c_'.. opt.class .. '_'.. i
	end
	if i>0 then
		torch.save(mdl_name, {dqn = dqn, gpu = opt.gpu})
	end
	logger:style{'+-'}
	logger:plot()
end













