require 'torch'
require 'math'

local max_mask = 256

function func_mask_random_init(...)
	local arg={...}
	local total_frms = arg[1] 
	-- generate a ramdom mask longger than 16 frames
	-- because C3D need at least 16 frames
	local n1 = 0
	local n2 = 0
	local generator = torch.Generator()
	if total_frms - torch.floor(max_mask/4) <= 5 then
		return {1, total_frms}
	else
		n1 = torch.random(generator,1,total_frms - torch.floor(max_mask/4) )
		n2 = n1 + torch.floor(max_mask/4)
	end
	if #arg == 1 then
		return {n1, n2}
	else
		-- do not see masked area
		local mask_table = arg[2]
		local flag = true
		local count = 0
		while flag and count < 10
		do
			flag = false
			for i=1,#mask_table 
			do
				if (n1 > mask_table[i][1] or n1 < mask_table[i][2]) or 
						(n2 > mask_table[i][1] or n2 < mask_table[i][2]) then
					n1 = torch.random(generator,1,total_frms- torch.floor(max_mask/4))
					n2 = n1+ torch.floor(max_mask/4)
					flag = true
					break
				end
			end
			count = count+1
		end -- while flag
		return {n1, n2}
	end -- if #arg == 1

end

--[[
function func_update_available_and_masked_segs(available, masked)
	local tmp_mask = masked[#masked]
	
	for i,v in pairs(available)
	do
		if tmp_mask[1] >= v[1] and tmp_mask[2] <= v[2] then
			local left = {v[1], tmp_mask[1]-1}
			local right = {tmp_mask[2]+1, v[2]}
			
			if left[2]-left[1] >= 15 then
				table.insert(available, left) -- insert the new left to available
			else
				masked[#masked][1] = left[1] -- merge the new left to masked
			end
			
			if right[2]-rightt[1] >= 15 then
				table.insert(available, right) -- insert the new right to available
			else
				masked[#masked][2] = right[2] -- merge the new right to masked
			end
			
			break
		end
	end
	table.remove(available,i)
	
	return available, masked
end
]]--

function func_take_action(old_mask, action, total_frms,alpha)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 4 then -- expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha / 2)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		
		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 16 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 16) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 5	then -- jump	
		local generator = torch.Generator()
		new_mask[1] = torch.random(generator,1,total_frms - (old_mask[2]-old_mask[1]) )
		new_mask[2] = new_mask[1]+(old_mask[2]-old_mask[1])
	else
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) < 16 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end

-- temporary for validate
function func_take_action_forward(old_mask, action, total_frms,alpha)
	
	local new_mask = {}
	local len = 0	
	local offset = 0
	
	if action == 1 then -- move forward
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
			new_mask[2] = old_mask[2] - offset
		else
			new_mask[1] = 1
			new_mask[2] = len
		end 
		
	elseif action == 2 then -- move back
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end 
		
	elseif action == 4 then -- expand
		len = old_mask[2] - old_mask[1] + 1
		if len > max_mask then
			return old_mask
		end
		offset = torch.floor(len * alpha / 2)
		
		if (old_mask[1] - offset) > 0 then
			new_mask[1] = old_mask[1] - offset
		else
			new_mask[1] = 1
		end 
		
		if (old_mask[2] + offset) < total_frms  then
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[2] = total_frms
		end
		
	elseif action == 3	then -- narrow
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha / 2)
		
		-- check if at least has 16 frames
		if (len - 2*offset) >= 16 then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		else
			offset = torch.floor((len - 16) / 2)
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] - offset
		end
	elseif action == 5	then -- jump -> take a giant leap instead
		len = old_mask[2] - old_mask[1] + 1
		offset = torch.floor(len * alpha * 3)
		
		if (old_mask[2] + offset) < total_frms then
			new_mask[1] = old_mask[1] + offset
			new_mask[2] = old_mask[2] + offset
		else
			new_mask[1] = total_frms - len + 1
			new_mask[2] = total_frms
		end  
	else
		error('Wrong action')
	end 
	if (new_mask[2] - new_mask[1]+1) < 16 then
			print(new_mask[1], new_mask[2])
			error('inadequate frames action' .. action)
	end
	return new_mask
end
