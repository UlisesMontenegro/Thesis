function n = neighbors(cell, map_dimensions)

  n = [];

  pos_x = cell(2);
  pos_y = cell(1);
  size_x = map_dimensions(2);
  size_y = map_dimensions(1);
  
  %%% YOUR CODE FOR CALCULATING THE NEIGHBORS OF A CELL GOES HERE
  
  % Return nx2 vector with the cell coordinates of the neighbors. 
  % Because planning_framework.m defines the cell positions as pos = [cell_y, cell_x],
  % make sure to return the neighbors as [n1_y, n1_x; n2_y, n2_x; ... ]
  for i = -1:1
      for j = -1:1
          if((((pos_y+i)>=1) && ((pos_y+i)<=size_y)) && (((pos_x+j)>=1) && ((pos_x+j)<=size_x)) && ((i~=0)||(j~=0)))
              %Add neighbor
              n = [n;[pos_y+i,pos_x+j]];
          end
      end
  end  

end