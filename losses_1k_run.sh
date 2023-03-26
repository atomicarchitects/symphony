for i in 1 2 3 4;
    do
    for l in 0 1 2 3 4 5;
        do python losses_1k.py $i $l 32;
        sleep 1;
    done
done