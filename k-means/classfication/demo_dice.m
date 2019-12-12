for i = 1 : size(dice,3)
    for j = 1 : size(dice,4)
        temp = dice(:,31:60,i,j);
        result(i,j) = mean(temp(:));
        result2(i,j) = std(temp(:));
    end
end
result
result2

