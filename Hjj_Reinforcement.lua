require 'torch'
require 'nn'

--different actions for the agent
number_of_actions = 6
--actions captures in the history vector
history_action_buffer_size = 4
--C3D size 
C3D_size = 4096 
--reward for middle stage actions
local reward_middle_action = 1
--reward fot terminal actions
local reward_terminal_action = 3
--IoU required to consider a positive detection
local iou_thd = 0.5

function func_modify_gt(gt_table, max_length)
	local new_gt = {}
	for i,p in pairs(gt_table) 
	do
		-- clips
		local tmp_gt = {}
		for j, q in pairs(p)
		do
			local tmp_max_length = max_length
			while tmp_max_length >= 96
			do
				if q[2] - q[1] <= max_length
				then
					if q[1] == 0 then q[1] = 1 end
					table.insert(tmp_gt, q)
				else
					if q[1] == 0 then q[1] = 1 end
					local beg = q[1]
					local ed = beg+max_length-1
					local step = torch.floor(max_length/3)
					table.insert(tmp_gt, {beg, ed, q[3]})
					while( ed+step <= q[2] )
					do
						beg = beg+step
						ed = ed+step
						table.insert(tmp_gt, {beg, ed, q[3]})
					end
					table.insert(tmp_gt, {beg+step, q[2], q[3]})
					
				end
				tmp_max_length = tmp_max_length - 32
			end
		end
		table.insert(new_gt,tmp_gt)
	end
	return new_gt
end

function func_update_history_vector(history_vector, action)
	local updated_history_vector = torch.Tensor(number_of_actions*history_action_buffer_size):zero()
	local stored_action_number = history_vector:nonzero():numel()
	
	if history_action_buffer_size > stored_action_number then
		updated_history_vector = history_vector:clone()
		updated_history_vector[stored_action_number*number_of_actions + action] = 1 
	else
		updated_history_vector[{ {1,number_of_actions*(history_action_buffer_size-1)} }] = 
			history_vector[{ {number_of_actions+1, -1} }]
		updated_history_vector[-(number_of_actions-action+1)] = 1
	end	
	
	return updated_history_vector
end 


function func_get_reward_movement(iou, new_iou, dist, new_dist)
	
	if new_iou > iou then
		return	reward_middle_action
	elseif new_iou < iou then
		return -reward_middle_action
	elseif new_iou == 0 then -- new_iou == iou == 0
		if new_dist < dist then
			return	reward_middle_action
		else
			return -reward_middle_action
		end
	else -- new_iou == iou ~= 0
		return -reward_middle_action
	end	
end

function func_get_reward_trigger(new_iou)
	if new_iou > iou_thd then
		return	reward_terminal_action
	else
		return -reward_terminal_action
	end	
end

--create a new dqn
function func_create_dqn()
	local feature_dim = C3D_size+number_of_actions*history_action_buffer_size
	local hid_dim = 1024
	local net = nn.Sequential()
	net:add(nn.Reshape(feature_dim))
	net:add(nn.Linear(feature_dim, hid_dim))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim, hid_dim))
	net:add(nn.ReLU())
	net:add(nn.Dropout(0.2))
	net:add(nn.Linear(hid_dim, number_of_actions))
	return net
end

--create/load dqn for specific class
function func_get_dqn(name, gpu, file)
	if name == '0' then
		file:write('Creating new DQN...\n')
		return func_create_dqn(), gpu
	else
		file:write('Loading saved DQN ' .. name .. ' ...\n')
		local obj = torch.load(name)
		return obj.dqn, obj.gpu -- load dqn
	end
end

-- sample data for minibatch
function func_sample(data, batch_size) 
	local minibatch = {}
	for i = 1, batch_size 
	do
		table.insert(minibatch,data[torch.random(
					torch.Generator(),1, table.getn(data))])
	end
	return minibatch
end




