function heur = heuristic(cell, goal)
  
  heur = 0;
  
  %%% YOUR CODE FOR CALCULATING THE REMAINING COST FROM A CELL TO THE GOAL GOES HERE
  heur = floor(pdist([cell;goal],'euclidean'));
  
end
