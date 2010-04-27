#! /lusr/bin/bash

cd out
a=`cat atts.txt`
convert img.jpg -fill red -stroke black -pointsize 14 -gravity NorthWest -annotate 0 "$a" final.jpg

if [ -f cur.jpg ]; then
    rm cur.jpg
fi

for i in 1 2 3 4 5; do
    for j in `ls | egrep '.*img[0-9]{'$i'}.jpg'`
      do
      echo $j
      f=${j/img/atts}
      f=${f/jpg/txt}

      a=`cat $f`
      convert $j -fill red -stroke black -pointsize 16 -gravity NorthWest -annotate 0 "$a" test.jpg

      if [ -f cur.jpg ]; then
	  convert +append cur.jpg test.jpg cur.jpg
      else
	  mv test.jpg cur.jpg
      fi

    done

if [ -f cur.jpg ]; then
    convert -append final.jpg cur.jpg final.jpg
    rm cur.jpg
fi
done

cd ..
