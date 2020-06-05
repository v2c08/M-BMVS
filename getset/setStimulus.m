function setStimulus(num)
    global stim
    stim = im2double(imread(num));
    if size(stim,3) == 3
        stim = rgb2gray(stim);
    end
    if size(stim,1) == 28
        stim = padarray(stim,[2 2],0,'both');
    end
    assert(size(stim,1)==32);
end