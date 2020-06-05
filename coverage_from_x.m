function im_sq  = coverage_from_x(im_dim,fov_width,x)
% Samples stimulus at a foveation location 

% im_v   - 2d stimulus
% im_dim - stimulus size 
% fov width - size of moving window
% x - Sampling location

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
 
im_sq = zeros(fov_width,fov_width);% make vectorised image square

for i = 1:length(xxx)
    for j = 1:length(yyy)
        im_sq(i,j) = sub2ind([32, 32], xxx(i), yyy(j));
    end
end
