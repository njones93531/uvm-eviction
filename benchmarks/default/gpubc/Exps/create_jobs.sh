nb=1000
xpar=4
j=4
#for i in `cat /home/aerdem/big_for_bc/big_list`
##for i in `cat /home/aerdem/big_for_bc/new_big_list`
#do
#    for deg1 in 0 1
#       do
#            for ord in 0 1
#            do
#                for het in 0 1
#                do
#                    id=${i#*_bc/}
#                    name=$id$"_"$ord$"_"$deg1$"_"$j$"___"$het       
#                    echo "#!/bin/bash" >tmp
#                    echo "#PBS -N "$name$".result" >>tmp
#                    echo "#PBS -l nodes=1:ppn=8:gpu" >>tmp
#                    echo "#PBS -l walltime=48:00:00" >>tmp
#                    echo "cd /home/aerdem/eriks_gpu_bc/gpu_bc/Exps" >>tmp
#                    echo "program=/home/aerdem/eriks_gpu_bc/gpu_bc/Exps/execs/gpu_bc" >>tmp
#                    echo "date" >>tmp
#                    if [ $het -eq 1 ] 
#                    then
#                        echo "OMP_NUM_THREADS=8 \$program "$i$" "$ord$" "$deg1$" "$j$" "$nb$" "$xpar >>tmp
#                    else
#                        echo "OMP_NUM_THREADS=1 \$program "$i$" "$ord$" "$deg1$" "$j$" "$nb$" "$xpar >>tmp
#                    fi
#                    echo "date" >>tmp
#                    mv tmp $name$".job"
#                done
#            done
#        done
#    done


#for i in `cat /home/aerdem/big_for_bc/big_list`
for i in `cat /home/aerdem/big_for_bc/new_big_list`
do
    for j in 2 3 4
    do
        for deg1 in 0 1
            do
                for ord in 0 1
                do
                    for het in 0 1
                    do
                        id=${i#*_bc/}
                        name=$id$"_"$ord$"_"$deg1$"_"$j$"___"$het       
                        echo "#!/bin/bash" >tmp
                        echo "#PBS -N "$name$".result" >>tmp
                        echo "#PBS -l nodes=1:ppn=8:gpu" >>tmp
                        echo "#PBS -l walltime=48:00:00" >>tmp
                        echo "cd /home/aerdem/eriks_gpu_bc/gpu_bc/Exps" >>tmp
                        echo "program=/home/aerdem/eriks_gpu_bc/gpu_bc/Exps/execs/gpu_bc" >>tmp
                        echo "date" >>tmp
                        if [ $het -eq 1 ] 
                        then
                            echo "OMP_NUM_THREADS=8 \$program "$i$" "$ord$" "$deg1$" "$j$" "$nb$" "$xpar >>tmp
                        else
                            echo "OMP_NUM_THREADS=1 \$program "$i$" "$ord$" "$deg1$" "$j$" "$nb$" "$xpar >>tmp
                        fi
                        echo "date" >>tmp
                        mv tmp $name$".job"
                    done
                done
            done
        done
    done






