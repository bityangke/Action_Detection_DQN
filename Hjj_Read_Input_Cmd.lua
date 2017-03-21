require 'torch'

function func_read_expert_cmd(cmd,arg)
	cmd:text()
	cmd:text('Create Expert Experience:')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-data_path', '', 'dataset path or expert memory path')
	cmd:option('-load_memory', 0, 'load saved expert memory instead of create new ones')
	cmd:option('-class', 0, 'The class you want to train')
	cmd:option('-model_name', '0', 'Name of the pretrained model to load')
	cmd:option('-gpu', -1, 'gpu enable flag, input is the gpu id,')
	cmd:option('-batch_size', 100, 'batch size, default')
	cmd:option('-lr', 1e-6, 'learning rate, default')
	cmd:option('-epochs', 10, 'epochs, default')
	
	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt
end

function func_read_training_cmd(cmd,arg)
	cmd:text()
	cmd:text('Train Agent:')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-data_path', '', 'Training dataset path')
	cmd:option('-name', 'a', 'name to save model')
	cmd:option('-class', 0, 'The class you want to train')
	cmd:option('-model_name', '0', 'Name of the pretrained model to load')
	cmd:option('-gpu', -1, 'gpu enable flag, input is the gpu id,')
	cmd:option('-alpha', 0.2, 'action scalar, default')
	cmd:option('-log_err','./log/training_error.log', 'log training error file')
	cmd:option('-log_log', './log/log', 'log file')
	cmd:option('-batch_size', 100, 'batch size, default')
	cmd:option('-replay_buffer', 1000, 'experience replay memory size, default')
	cmd:option('-lr', 1e-6, 'learning rate, default')
	cmd:option('-epochs', 50, 'epochs, default')
	
	
	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt
end

function func_read_validate_cmd(cmd, arg)
	cmd:text()
	cmd:text('Validate Agent:')
	cmd:text()
	cmd:text('Options:')
	
	cmd:option('-data_path', '', 'Training dataset path')
	cmd:option('-class', 0, 'The class you want to train')
	cmd:option('-model_name', '0', 'Name of the pretrained model to load')
	cmd:option('-gpu', -1, 'gpu enable flag, input is the gpu id,')
	cmd:option('-alpha', 0.2, 'action scalar, default')
	cmd:option('-log_log', './log/v_log', 'log file')
	cmd:option('-max_steps', 50, 'max step for one clip, default ')
	
	cmd:text()
	
	local opt = cmd:parse(arg)
	
	return opt
end


function func_set_gpu(opt, file)
	if opt >= 0 then
		require 'cutorch'
		require 'cunn'
		if opt == 0 then
			local gpu_id = tonumber(os.getenv('GPU_ID'))
			if gpu_id then 
				opt = gpu_id+1 
			end
		end
		if opt > 0 then 
			cutorch.setDevice(opt) 
		end
		opt = cutorch.getDevice()
		file:write('Using GPU device id:'.. opt-1 .. '\n')
		print('Using GPU device id:'.. opt-1)
	else
		file:write('Using CPU code only. GPU device id:' .. opt .. '\n')
		print('Using CPU code only. GPU device id:' .. opt)
	end
	return opt
end
