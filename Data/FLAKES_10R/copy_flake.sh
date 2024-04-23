for i in `ls ../FLAKES_10?_600$1/sp_*/*_*e_$1-*.pdb`; do
    echo $i
    cp $i .
    cat ../FLAKES_10?_600$1/00-System.atoms | sed /#/d | gawk -v a=$1 '{print $0 a}' >> 00-System.ATOMS
done
