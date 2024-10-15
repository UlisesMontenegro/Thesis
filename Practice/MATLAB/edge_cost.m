function cost = edge_cost(parent, child, map)

  cost = 0;
  threshold = 0.1;
 
  %%% YOUR CODE FOR CALCULATING THE COST FROM VERTEX parent TO VERTEX child GOES HERE
  if map(child(1),child(2)) > threshold
      cost = inf
  else
      cost = 1
      %cost = map(child(1),child(2))*100
end