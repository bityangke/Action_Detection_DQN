-- split to detect
-- combine the test result to calculate mAP 

require 'Hjj_Read_Input_Cmd'
require 'Hjj_Reinforcement'
require 'Zt_Interface_new'
require 'Hjj_Mask_and_Actions'
require 'Hjj_Metrics'

local cmd = torch.CmdLine()
opt = func_read_validate_cmd(cmd, arg)

-- create log file
local log_file = io.open(opt.log_log, 'w')
if not log_file then
	print("open log file error")
	error("open log file error")
end

local track_file = io.open('./data_output/tracka.txt', 'w')
if not track_file then
	print("open track file error")
	error("open track file error")
end

local gt_file = io.open('./data_output/gta.txt', 'w')
if not gt_file then
	print("open gt file error")
	error("open gt file error")
end

-- read validate clips from files
--local validate_file = './' .. opt.data_path .. '/validationlist_id.t7'
--local validate_file = './' .. opt.data_path .. '/trainlist_id.t7'
local validate_file = './' .. opt.data_path .. '/new_validatelist_thumos.t7'
print(validate_file)
local clip_table = torch.load(validate_file)
local tt = clip_table[opt.class]
if tt == nil then
	error('no trainlist file')
end

-- thomas
local validate_clip_table={}
--validate_clip_table = tt
for i=1,10 do
	table.insert(validate_clip_table, tt[#tt-10+i])
end

-- action parameters
local max_steps = opt.max_steps
local trigger_thd = 0.5 -- threshold for terminal
local trigger_action = number_of_actions
local act_alpha = opt.alpha
local max_trigger = 11
local mask_rate = 0.05 -- 1-2*mask_rate of current mask will not be used anymore

-- number_of_actions and history_action_buffer_size are globle variables in Hjj_Reinforcement
local history_vector_size = number_of_actions * history_action_buffer_size
local input_vector_size = history_vector_size + C3D_size

-- load dqn
if opt.model_name == '0' then
	error('model needed')
end
local dqn={}
dqn, opt.gpu = func_get_dqn(opt.model_name, opt.gpu, log_file)

-- set gpu
opt.gpu = func_set_gpu(opt.gpu, log_file)
if opt.gpu >=0 then dqn = dqn:cuda() end

local gt_table = func_get_data_set_info(opt.data_path, opt.class, 1)
local max_gt_length = 128 -- max length to split gt
--gt_table = func_modify_gt(gt_table, max_gt_length)

-- load C3D model
local C3D_m = torch.load('c3d.t7');

-- used to visualize the action sequence
local iou_record_table = {}
local gt_ind_record_table = {}
local mask_record_table= {}


for i,v in pairs(validate_clip_table)
do
	-- split clips
		local trigger_count = 0 
		local tmp_gt = gt_table[v]

		for c=1,#tmp_gt do
			gt_file:write(i .. '\t' .. tmp_gt[c][1] .. '\t' .. tmp_gt[c][2] .. '\n')
		end
	
		local total_frms = tmp_gt[1][3]
		local masked_segs={}
		local gt_num = table.getn(tmp_gt)
		local steps = 0

		log_file:write('\tIt is the ' .. i .. ' clip, clip_id = ' .. 
							v .. ' total_frms = '.. total_frms .. '\n')
		print('\tIt is the '.. i .. ' clip, clip_id = ' .. 
							v .. ' total_frms = '.. total_frms)
		local lp=1
		local left_frm = total_frms
		local knocked = 0
		local last_f = 1					
		--while (steps < max_steps) and (trigger_count < max_trigger)
		while (left_frm > 16) and (knocked < 5)
		do
			local iou = 0
			local index = 0
			--local mask = func_mask_random_init(total_frms, masked_segs)
			-- continue from the end of last mask
			local mask = {last_f, last_f+64}
			if last_f - 16 > 0 then 
				mask[1] = mask[1] - 16
				mask[2] = mask[2] - 16
			 end
			
			if mask[2] >= total_frms then 
				mask[2] = total_frms
				knocked = knocked + 1
			end
			
			local history_vector = torch.Tensor(history_vector_size):fill(0) 
			local bingo = false
			local action = 0
		
			--while (steps < max_steps) and (not bingo)
			while (left_frm > 16) and (knocked < 5) and (not bingo)
			do
				iou, index = func_find_max_iou(mask, tmp_gt)
			
				track_file:write(i .. '\t' .. lp .. '\t' .. steps .. '\t' ..
								 iou .. '\t' .. mask[1] .. '\t' .. mask[2] .. '\t' .. action .. '\n')
			
				print('\t\tstep ' .. steps .. '\t; beg = ' .. mask[1] .. '\t ;end = ' .. mask[2] 
								.. ' ; iou ' .. iou .. '\t' .. action .. '\n')
												
				local C3D_vector = func_get_C3D(opt.data_path, opt.class, 1,
												 v, mask[1], mask[2], C3D_m, {})
				local input_vector = torch.cat(C3D_vector, history_vector, 1)
			
				if opt.gpu >=0 then input_vector = input_vector:cuda() end
			
				local action_output = dqn:forward(input_vector)
				local tmp_v = 0
				
				tmp_v, action = torch.max(action_output,1)
				action = action[1]-- from tensor to numeric type
				-- give a very small number for getting the second max action
				action_output[action] = -111111111 
				
				print('\t\t\tAction = ' .. action .. '\n')
				
				
				if action == 3 and mask[2]-mask[1] <= 16 then
					tmp_v, action = torch.max(action_output,1)
					action = action[1]-- from tensor to numeric type
				elseif action == 4 and mask[2]-mask[1]+1 >= 2*max_gt_length then
					tmp_v, action = torch.max(action_output,1)
					action = action[1]-- from tensor to numeric type
				end
				if action == trigger_action then
					print('############### BOOM! #############'.. mask[1] .. 
						' - ' .. mask[2] .. ' ; ' .. total_frms .. '\n')
					track_file:write(i .. '\t' .. lp .. '\t' .. steps .. '\t' ..
								 iou .. '\t' .. mask[1] .. '\t' .. mask[2] .. '\t' .. action .. '\n')
					bingo = true
					trigger_count = trigger_count + 1
					-- not to mask all the area
					if last_f < mask[2] then
						last_f = mask[2]
						left_frm = total_frms - last_f
					end
					if mask[2] == total_frms then
						knocked = knocked + 1
					end
					mask[1] = mask[1]+torch.floor((mask[2]-mask[1])*mask_rate)
					mask[2] = mask[2]-torch.floor((mask[2]-mask[1])*mask_rate)
					table.insert(masked_segs, mask)
				else
					mask = func_take_action_forward(mask, action, total_frms, act_alpha)
					if last_f < mask[2] then
						last_f = mask[2]
						left_frm = total_frms - last_f
					elseif last_f - mask[2] >= 64 then
						bingo = true
						print('~~~~~~go back too much!!!!')
					end
					if mask[2] == total_frms then
						knocked = knocked + 1
					end
				end
				history_vector = func_update_history_vector(history_vector, action)
				steps = steps + 1
			end	--while
			lp = lp+1
		end -- while
	
end --for clips

gt_file:close()
track_file:close()















