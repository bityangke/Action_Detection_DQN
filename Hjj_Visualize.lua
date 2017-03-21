require 'Hjj_Read_Input_Cmd'
require 'torch'
require 'gnuplot'

--read input para
local cmd = torch.CmdLine()
opt = func_read_visualize_cmd(cmd, arg)

local gt_ind_record_table, iou_record_table, mask_record_table = torch.load(opt.name)

for i=1, #gt_ind_record_table
do
	local name1 = './plot/plot_iou_'.. i .. '.png'
	local name2 = './plotplot_move_' .. i .. '.png'
	
	local tmp_gt_record = gt_ind_record_table[i]
	local tmp_iou_record = iou_record_table[i]
	local tmp_mask_record = mask_record_table[i]
	
	local flag = true
	local count = 1
	local iou1 = {}
	local iou2 = {}
	
	
	for j=1, #tmp_iou_record
	do
		if flag then
			for k, v in pairs(tmp_iou_record[j])
			do
				table.insert(iou1, {v, count})
				count = count+1
			end
			flag = false
		else
			for k, v in pairs(tmp_iou_record[j])
			do
				table.insert(iou2, {v, count})
				count = count+1
			end
			flag = true
		end
	end
	
	local iou1_t = torch.Tensor(#iou1):fill(0)
	local idx1_t = torch.Tensor(#iou1):fill(0)
	for j, v in pairs(iou1) 
	do
		iou1_t[j] = v[1]
		idx1_t[j] = v[2]
	end
	
	local iou2_t = torch.Tensor(#iou2):fill(0)
	local idx2_t = torch.Tensor(#iou2):fill(0)
	for j, v in pairs(iou2) 
	do
		iou2_t[j] = v[1]
		idx2_t[j] = v[2]
	end
	gnuplot.pngfigure(name1)
	gnuplot.plot(
				{idx1_t, iou1_t, '*'},{idx2_t, iou2_t, '+'})
	gnuplot.plotflush()
	
	for j=1, #tmp_iou_record
	do
		if flag then
			for k, v in pairs(tmp_iou_record[j])
			do
				table.insert(iou1, {v, count})
				count = count+1
			end
			flag = false
		else
			for k, v in pairs(tmp_iou_record[j])
			do
				table.insert(iou2, {v, count})
				count = count+1
			end
			flag = true
		end
	end
	
	local iou1_t = torch.Tensor(#iou1):fill(0)
	local idx1_t = torch.Tensor(#iou1):fill(0)
	for j, v in pairs(iou1) 
	do
		iou1_t[j] = v[1]
		idx1_t[j] = v[2]
	end
	
	local iou2_t = torch.Tensor(#iou2):fill(0)
	local idx2_t = torch.Tensor(#iou2):fill(0)
	for j, v in pairs(iou2) 
	do
		iou2_t[j] = v[1]
		idx2_t[j] = v[2]
	end
	gnuplot.pngfigure(name1)
	gnuplot.plot(
				{idx1_t, iou1_t, '*'},{idx2_t, iou2_t, '+'})
	gnuplot.plotflush()
	
end









