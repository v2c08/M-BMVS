function fov    = im_centred_on_x(im_v,im_dim,fov_width,x)
% Foveated Sampling
% x         - pixel values centered x 
% im_v      - stimulus
% im_dim    - size of stimulus (32*32)
% fov width - size of moving window
% x         - sampling location, x should be in the range 0 to 1


x(x <0) = 0; x(x>1) = 1;

xx = linspace(0,1,im_dim); % closest range for [xx xy]
xy = linspace(0,1,im_dim); % closest range for [xx xy]

pixel_xx = find(abs(xx-x(1)) == min(abs(xx-x(1))));
pixel_xy = find(abs(xy-x(2)) == min(abs(xy-x(2))));

% checks on edges X dimension
if pixel_xx < ceil(fov_width/2)
    pixel_xx =  fov_width/2;
    xxx      =  (pixel_xx+1) -fov_width/2:pixel_xx+(fov_width/2);
elseif pixel_xx > floor(im_dim - fov_width/2)
    pixel_xx = im_dim -fov_width/2;
    xxx      =  ((pixel_xx) -fov_width/2 +1):pixel_xx+(fov_width/2);
else
     xxx      =  (pixel_xx+1) -fov_width/2:pixel_xx+(fov_width/2);
end

% checks on edges Y dimension
if pixel_xy < ceil(fov_width/2)
    pixel_xy =  fov_width/2;
    yyy      =  (pixel_xy+1) -fov_width/2:pixel_xy+(fov_width/2);
elseif pixel_xy > floor(im_dim - fov_width/2)
    pixel_xy = im_dim -fov_width/2;
    yyy      =  ((pixel_xy) -fov_width/2 +1):pixel_xy+(fov_width/2);
else
      yyy      =  (pixel_xy+1) -fov_width/2:pixel_xy+(fov_width/2);
end

im_sq = spm_unvec(im_v,zeros(im_dim,im_dim));% make vectorised image square
fov   = im_sq(xxx,yyy);
 