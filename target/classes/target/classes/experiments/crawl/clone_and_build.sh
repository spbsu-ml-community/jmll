git clone git@github.com:crawl/crawl.git
cd crawl
git submodule update --init
cd crawl-ref/source
make WEBTILES=y -j8

