function map = convolve_map(map, times)
    for t = 1:times
        for i = 1:size(map,1)
            for j = 1:size(map,2)
                if(i == 1)
                    map(i,j) = (2/3)*map(i,j) + (1/3)*map(i+1,j);
                elseif(i == size(map,1)-1)
                    map(i,j) = (1/3)*map(i-1,j) + (2/3)*map(i,j);
                elseif((i>1)&&(i<(size(map,1)-1)))
                    map(i,j) = 0.25*map(i-1,j) + 0.5*map(i,j) + 0.25*map(i+1,j);
                end
            end
        end
    end
end