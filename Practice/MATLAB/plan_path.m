function inverted_path = plan_path(map2, start, goal)
    % load the map
    %load mapa_fiuba_1p.mat      %carga el mapa como occupancyMap en la variable 'map'
    % load mapa_lae.mat         %mapa viejo para probar cosas

    
    %if verMatlab.Release=='(R2019a)'
        %Para versiones anteriores de MATLAB, puede ser necesario ajustar mapa
        %imagen_mapa = 1-double(imread('mapa_fiuba_1p.tiff'))/255;
        %map2 = robotics.OccupancyGrid(imagen_mapa, 25);
    %elseif verMatlab.Release(1:5)=='(R201'    % Completar con la version que tengan
        %Ni idea que pasa, ver si el truco R2016b funciona
        %disp('ver si la compatibilidad R2016b funciona');
    %else
        %disp(['Utilizando MATLAB ', verMatlab.Release]);
    %end
   
    
   %Convert map into occupancy matrix and convolve it to avoid collisions
    map = occupancyMatrix(map2);
    map = convolve_map(map,20);
    
   %start = [8.3 7];
   %goal = [5.3 4.3];
    
    % NOTE that in octave/MATLAB, matrices are accessed with A(y,x),
    % where y is the row and x is the column

    % visualizes the map. Note: 
    % the y-axis is inverted, so 0,0 is top-left, 
    figure(1)
    imshow(map);
    title ('Map');
    hold on;

    inverted_path = [];
    % retrieve height and width of the map
    [h,w] = size(map);

    % cost values for each cell, filled incrementally. Initialize with infinity
    costs = ones(h,w)*inf;

    % estimated costs to the goal.
    heuristics = zeros(h,w);

    % cells that have been visited
    closed_list = zeros(h,w);

    % these matrices implicitly store the path
    % by containing the x and y position of the previous
    % node, respectively. Following these starting at the goal 
    % until -1 is reached returns the computed path, see at the bottom
    previous_x = zeros(h,w)-1;
    previous_y = zeros(h,w)-1;

    % start and goal position (y,x)
    start = world2grid(map2, start);
    goal = world2grid(map2, goal);

    %plot start and goal cell
    plot(start(2), start(1), 'r.');
    plot(goal(2), goal(1), 'g.');
    pause(1); %pause for visualization

    %start search at the start 
    parent=start;
    costs(start(1),start(2)) = 0;

    %loop until the goal is found
    while (parent(1)~=goal(1) || parent(2)~=goal(2))

      %generate mask to assign infinite costs for cells already visited
      closed_mask = closed_list;
      closed_mask( closed_mask==1 )=Inf; 

      %find the candidates for expansion (open list/frontier)
      open_list = costs + closed_mask + heuristics;

      %check if a non-infinite entry exists in open list (list is not empty)
      if min(open_list(:))==Inf
        disp('no valid path found');
        break
      end

      %find the cell with the minimum cost in the open list
      [y,x] = find(open_list == min(open_list(:)));
      parent_y = y(1);
      parent_x = x(1);
      parent = [parent_y, parent_x];

      %put parent in closed list
      closed_list(parent_y,parent_x) = 1;
      plot(parent_x, parent_y, ' y.' );

      %for visualization: Plot start again
      if(parent(1) == start(1) && parent(2) == start(2))
        plot(start(2), start(1), 'r.');
      end

      %get neighbors of parent
      n = neighbors(parent, [h,w]);
      for i=1:size(n,1)
        child_y = n(i,1);
        child_x = n(i,2);
        child = [child_y, child_x];

        %calculate the cost of reaching the cell
        cost_val = costs(parent_y,parent_x) + edge_cost(parent, child, map);

        %Estimate the remaining costs from the cell to the goal
        heuristic_val = heuristic(child, goal);

        %update cost of cell
        if cost_val < costs(child_y,child_x)
          costs(child_y,child_x) = cost_val;
          heuristics(child_y,child_x) = 2*heuristic_val;

          %safe child's parent
          previous_x(child_y,child_x) = parent_x;
          previous_y(child_y,child_x) = parent_y;
        end
      end
      pause(0.05); %pause for visualization
    end

    % visualization: from the goal to the start,
    % draw the path as blue dots
    parent = [goal(1), goal(2)];
    distance2 = 0;
    while previous_x(parent(1), parent(2))>=0
      plot(parent(2), parent(1), 'b.');
      inverted_path = [inverted_path; grid2world(map2, [parent(1) parent(2)])];

      %for visualization: Plot goal again
      if(parent(1) == goal(1) && parent(2) == goal(2))
        plot(goal(2), goal(1), 'g.');
      end

      child_y = previous_y(parent(1), parent(2));
      child_x = previous_x(parent(1), parent(2));
      child = [child_y, child_x];
      distance2 = distance2+norm(parent - child);
      parent = child;
      pause(0.05); %pause for visualization
    end

    disp 'Path planned succesfully'
    disp 'Path cost: ', disp(costs(goal(1),goal(2)));
    disp 'Path length: ', disp(distance2);
    disp 'Number of nodes visited: ', disp(sum(closed_list(:)));

%end