mkdir datasets/hyperpartisan
wget -P datasets/hyperpartisan/ https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip
wget -P datasets/hyperpartisan/ https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip
unzip datasets/hyperpartisan/articles-training-byarticle-20181122.zip -d datasets/hyperpartisan
unzip datasets/hyperpartisan/ground-truth-training-byarticle-20181122.zip -d datasets/hyperpartisan
rm datasets/hyperpartisan/*zip