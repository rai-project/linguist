/*
Detect programming language of source files.
Go port of GitHub Linguist: https://github.com/github/linguist

Prerequisites:

    go get github.com/jteeuwen/go-bindata/go-bindata

Installation:

    mkdir -p $GOPATH/src/github.com/rai-project/linguist/pkg
    git clone --depth=1 https://github.com/rai-project/linguist/pkg $GOPATH/src/github.com/rai-project/linguist/pkg
    go get -d github.com/rai-project/linguist/pkg
    cd $GOPATH/src/github.com/rai-project/linguist/pkg
    make
    l

Usage:

Please refer to the source code for the reference implementation at:

https://github.com/rai-project/linguist/pkg/tree/master/cmd/l


See also:

https://github.com/rai-project/linguist/pkg/tree/master/tokenizer
*/
package linguist
