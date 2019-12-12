% 

pathname = '/bmrNAS/people/yuxinh/DL_diffseg/DiffSeg-Data';

files = dir([pathname,'/mwu*']);
for i = 1 : length(files)
    %%unzip([pathname,'/', files(i).name(1:end-7)]);
    mkdir([pathname,'/', files(i).name, '/jpg'])
    load([pathname,'/', files(i).name, '/data.mat'])
    for j = 1 : size(imgs, 3)
        imwrite(imresize(squeeze(imgs(:,:,j,[1,3,4])),[128 128]),[pathname,'/', files(i).name, '/jpg/','im', num2str(j),'.jpeg'],'JPEG');
    end

end

%%
pathname = '/bmrNAS/people/yuxinh/DL_diffseg/MSSeg-Data';

files = dir([pathname,'/*Case*']);
for i = 1 : length(files)
    %%unzip([pathname,'/', files(i).name(1:end-7)]);
    mkdir([pathname,'/', files(i).name, '/jpg'])
    load([pathname,'/', files(i).name, '/data.mat'])
    imgs = img;
    for j = 1 : size(imgs, 3)
        imwrite(repmat(squeeze(imgs(:,:,j))/255,[1 1 3]),[pathname,'/', files(i).name, '/jpg/','im', num2str(j),'.jpeg'],'JPEG');
    end
end