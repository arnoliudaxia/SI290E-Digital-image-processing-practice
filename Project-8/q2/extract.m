
% extracts the center (cc,cr) and radius of the largest blob
function [x2,y2,width_x,width_y,cc,cr,flag]=extract(Imwork,Imback,index)%,fig1,fig2,fig3,fig15,index)
  
  x2 = 0;
  y2 = 0;
  width_x = 0;
  width_y = 0;
  cc = 0;
  cr = 0;
  flag = 0;
  [MR,MC,Dim] = size(Imback);

  % subtract background & select pixels with a big difference
  fore = zeros(MR,MC);          %image subtracktion
  fore = (abs(Imwork(:,:,1)-Imback(:,:,1)) > 20) ...
     | (abs(Imwork(:,:,2) - Imback(:,:,2)) > 20) ...
     | (abs(Imwork(:,:,3) - Imback(:,:,3)) > 20);  

  % Morphology Operation  erode to remove small noise
  %foremm = bwmorph(fore,'erode',2); %2 time

  % select largest object
  labeled = bwlabel(fore,4);
  stats = regionprops(labeled,['basic']);%basic mohem nist
  [N,W] = size(stats);
  if N < 1
    return   
  end

  % do bubble sort (large to small) on regions in case there are more than 1
  id = zeros(N);
  for i = 1 : N
    id(i) = i;
  end
  for i = 1 : N-1
    for j = i+1 : N
      if stats(i).Area < stats(j).Area
        tmp = stats(i);
        stats(i) = stats(j);
        stats(j) = tmp;
        tmp = id(i);
        id(i) = id(j);
        id(j) = tmp;
      end
    end
  end

  % make sure that there is at least 1 big region
  if stats(1).Area < 100 
    return
  end
  selected = (labeled==id(1));

  % get center of mass and radius of largest
  centroid = stats(1).Centroid;
  boundingbox = stats(1).BoundingBox;
  x2=boundingbox(1)
  y2=boundingbox(2)
  width_x=boundingbox(3)
  width_y=boundingbox(4)
  cc = centroid(1);
  cr = centroid(2);
  flag = 1;
  return