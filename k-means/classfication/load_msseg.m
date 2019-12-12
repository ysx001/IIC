pathname = '/Users/huyx11/Downloads/wLLR/project221';
addpath(genpath('/bmrNAS/people/yuxinh/DL_diffseg'))   
pathname = '/bmrNAS/people/yuxinh/DL_diffseg/MSSeg';
savepath = '/bmrNAS/people/yuxinh/DL_diffseg/MSSeg-Data';
files = dir([pathname,'/*.nii']);
for i = 1 : length(files)
    %%unzip([pathname,'/', files(i).name(1:end-7)]);
    nii = load_nii([pathname,'/', files(i).name]);
    img = nii.img;
    save([savepath, '/', files(i).name(1:end-4)],'img')
end



%%
pathname = '/bmrNAS/people/yuxinh/DL_diffseg/MSSeg-Data';
files = dir([pathname,'/*FLAIR*']);
for i = 1 : length(files)
    %%unzip([pathname,'/', files(i).name(1:end-7)]);
    name = files(i).name(1:end-10);
    mkdir([pathname,'/', name]);
    if exist([pathname, '/', name,'_lesion.mat'], 'file') == 2
        load([pathname, '/', name,'_lesion.mat'])
        mask = img;
        load([pathname, '/', files(i).name])
        save([pathname, '/', name,'/data'],'img','mask')
    else
        load([pathname, '/', files(i).name])
        save([pathname, '/', name,'/data'],'img')
    end
end

