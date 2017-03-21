

-- 1) Install Torch:
--        See http://torch.ch/docs/getting-started.html#_
-- 2) Install necessary packages:
--        $ luarocks install loadcaffe
--        $ luarocks install hdf5
-- 3) Download a model:
--        $ wget http://places2.csail.mit.edu/models/vgg16_places2.tar.gz
--        $ tar xzvf vgg_places2.tar.gz
-- 4) Put images inside 'images' directory 
-- 5) Modify this script's options below
-- 6) Run it
--        $ th extract.lua
-- 7) Read it
--       In 'features', there will be one HDF5 file per image
--       with the dataset 'feat' containing the feature

-- some options

--assert(image_dir:sub(image_dir:len()) ~= '/', 'image_dir should not end with /')

-- load dependencies
require 'cutorch'     -- CUDA tensors
require 'nn'          -- neural network package
require 'cudnn'       -- fast CUDA routines for neural networks
require 'paths'       -- utilities for reading directories 
require 'image'       -- reading/processing images
require 'xlua'        -- for progress bar
require 'math'


-- set GPU device
-- check which GPUs are free with 'nvidia-smi'
-- first GPU is #1, second is #2, ...
--cutorch.setDevice(gpu_device)
function func_get_data_set_info(data_path,class_id,flag)

--#######################################
	local action_class={'High jump','Cricket','Discus throw','Javelin throw','Paintball','Long jump',
	                     'Bungee jumping','Triple jump','Shot put','Dodgeball','Hammer throw',
	                     'Skateboarding','Doing motocross','Starting a campfire','Archery',
	                     'Playing kickball','Pole vault','Baton twirling','Camel ride','Croquet',
	                     'Curling','Doing a powerbomb','Hurling','Longboarding','Powerbocking','Rollerblading'}
	--action_class[1] = 'High jump'
--########################################
    local gt_table={}
    local clip_table={}
    local dir_path={}
    dir_path = data_path ..'/'.. action_class[class_id] ..'/'
    local file_clipnum = io.open(dir_path..'ClipsNum.txt',"r")        --the number of clips in class class_id
    if file_clipnum then
	    local n = file_clipnum:read("*n")                  --read the number of clips in class class_id
	    --print('n='..n)
	    io.close(file_clipnum)
	    if(flag==0) then                                   --test dataset has no groudtruth(not released),so return a empty table
                for i=1,n do
                    local Frame_file_test = io.open(dir_path..i..'/FrameNum.txt',"r")   --the file keep the name of frames of each clips
		    		local FrameNum_test = Frame_file_test:read("*n")
		    		io.close(Frame_file_test)
		    		gt_table[i] = {{0,0,FrameNum}}                    
                end
            print('error return')
			return gt_table
	    elseif(flag==1) then                               --training
		--dir_path = data_path..'/'..'train/'..class_id..'/'
		  for i=1,n do
			gt_path = dir_path..i..'/gt.txt'             --the file which keep the groundtruth of each clip
			--print(gt_path)
			--error()
			local file_gt = io.open(gt_path,"r")
			--local j=1
			if file_gt then
			   local Frame_file = io.open(dir_path..i..'/FrameNum.txt',"r")   --the file keep the name of frames of each clips
			   local FrameNum = Frame_file:read("*n")
			   io.close(Frame_file)
			   --while(true) do
		       local gt = {}
			   local file_gtnum=io.open(dir_path..i..'/NumOfGt.txt',"r")         --the file keep the number of groundtruth of each clips
		       local gtnum = file_gtnum:read("*n")
		       io.close(file_gtnum)
		       clip_table={}
			   for k=1,gtnum do
			        gt={}
		           --local temp=file_gt:read("*n")
		           gt[1]=file_gt:read("*n")
			       gt[2]=file_gt:read("*n")
		           gt[3]=FrameNum
		           clip_table[k]=gt
		     
			   end
		        io.close(file_gt)
			     -- end
			   gt_table[i]=clip_table
			else
			   gt_table[i]={{0,0,0}}
			end

		end
	end
   end
   --print(gt_table) 
   return gt_table
end


function func_get_C3D(data_path,class_id,flag,clip_ind,beg_ind,end_ind, c3d_m,cover_id)

--#######################################
	local action_class={'High jump','Cricket','Discus throw','Javelin throw','Paintball','Long jump',
	                     'Bungee jumping','Triple jump','Shot put','Dodgeball','Hammer throw',
	                     'Skateboarding','Doing motocross','Starting a campfire','Archery',
	                     'Playing kickball','Pole vault','Baton twirling','Camel ride','Croquet',
	                     'Curling','Doing a powerbomb','Hurling','Longboarding','Powerbocking','Rollerblading'}
--########################################

	-- loads
        --local image_dir = 'images'
	--local out_dir = 'features'
	--local prototxt = 'vgg16_places2/deploy.prototxt'
	--local caffemodel = 'vgg16_places2/vgg16_places2.caffemodel'
	local layer_to_extract = 24 --fc6 -- 39=fc8, 37=fc7, 31=pool5  
          --layer 25 is the dropout and layer 24 is Relu,while layer 23 is Linear

	--local batch_size = 2
	local Width = 112
	local Height = 112
    local Channels = 3
	local gpu_device = 1 
	local mean_image = {128, 128, 128}
	local ext = 't7' -- 'h5' or 't7'
    
	local model = c3d_m
	local Length = 16                 --this may be too small,we need a far more larger length

	--model:evaluate() -- turn on evaluation model (e.g., disable dropout)
	model:cuda() -- ship model to GPU

	--print(model) -- visualizes the model
	--print('extracting layer ' .. layer_to_extract)

	-- tensor to store RGB images on GPU 
	local input_images = torch.Tensor(Channels,Length,Width,Height)

	-- utility function to check if file exists
	--function file_exists(name)     
	--  local f=io.open(name,"r")
	--  if f~=nil then io.close(f) return true else return false end
	--end

	-- function to read image from disk, and do preprocessing
	-- necessary for caffe models
          --if(flag == 1) then
	   --    local path = data_path..'/train/'..tostring(class_id)..'/'..tostring(clip_ind)..'/'
         -- end
          --else then
      local path = data_path..'/'..action_class[class_id]..'/'..tostring(clip_ind)..'/'
      --print(path)
	  local data = torch.Tensor(Length,Channels,Width,Height)
	  for t = 1,Length do
	      local cover_flag = false;
	      step = math.floor((end_ind-beg_ind)/Length)      --the step length
	      --print(path..tostring(beg_ind+(t-1)*step..'.jpg'))
          local f = io.open(path..tostring(beg_ind+(t-1)*step..'.jpg'))
          if f then 
          	io.close(f)
          else 
          	print('jpg file not found!')
            return
          end
		  ------this for loop is to find out if this frame needs to be covered---------------------------
		  for z=1,#cover_id do
		     local frame_id = beg_ind+(t-1)*step;
			 if(frame_id >= cover_id[z][1] and frame_id <=cover_id[z][2]) then
				cover_flag = true         --this frame need to be covered
				break
			end 
		  end
		  --------------------------------------------------------------------------------------------
		  -------------------------------------------------------------------------------------------
		  local im = {}
		  if cover_flag then
		    im = torch.zeros(Channels,Width,Height)    ---all black
		  else
			  im = image.load(path..tostring(beg_ind+(t-1)*step)..'.jpg')                 -- read image
			  im = image.scale(im, Width, Height)  -- resize image
			  im = im * 255                                 -- change range to 0 and 255
			  im = im:index(1,torch.LongTensor{3,2,1})      -- change RBB --> BGR
		  -- subtract mean
			  for i=1,3 do
				im[{ i, {}, {} }]:add(-mean_image[i])       --normalization
			  end
		  end
	      data[t] = im
       end
	  input_data = data:permute(2,1,3,4)
          input_data = input_data:float()
          input_data = input_data:cuda()
 
	-- function to run feature extraction
	  -- do forward pass of model on the images
	  model:forward(input_data)

	  -- read the activations from the requested layer
	  local feat = model.modules[layer_to_extract].output

	  -- ship activations back to host memory
	  feat = feat:double()
	  return feat
	  -- save feature for item in batch
	 --[[ for i=1,size-1 do
	    -- make output directory if needed
	    paths.mkdir(paths.dirname(feat_paths[i+last_id-1]))

	    if ext == 'h5' then -- save hdf5 file
	      local hdf5_file = hdf5.open(feat_paths[i+last_id-1], 'w')
	      hdf5_file:write('feat', feat[i])
	      hdf5_file:close()

	    elseif ext == 't7' then -- save torch7 file
	      torch.save(feat_paths[i+last_id-1], feat[i])

	    else
	      assert(false, 'unknown filetype')
	    end
	  end]]

end


