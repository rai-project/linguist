# linguist

[![godoc reference](https://godoc.org/github.com/rai-project/linguist/pkg?status.png)](https://godoc.org/github.com/rai-project/linguist/pkg)

Go port of [github linguist](https://github.com/github/linguist).

Many thanks to [@petermattis](https://github.com/petermattis) for his initial work in laying the groundwork of creating this project, and especially for suggesting the use of naive Bayesian classification.

Thanks also to [@jbrukh](https://github.com/jbrukh) for [github.com/jbrukh/bayesian](https://github.com/jbrukh/bayesian)

# install

### prerequisites:

```
go get github.com/jteeuwen/go-bindata/go-bindata
```

```
mkdir -p $GOPATH/src/github.com/rai-project/linguist/pkg
git clone --depth=1 https://github.com/rai-project/linguist/pkg $GOPATH/src/github.com/rai-project/linguist/pkg
go get -d github.com/rai-project/linguist/pkg
cd $GOPATH/src/github.com/rai-project/linguist/pkg
make
l
```

## see also

[command-line reference implentation](cmd/l) which is documented separately

[tokenizer](tokenizer/tokenizer.go) | ([godoc reference](https://godoc.org/github.com/rai-project/linguist/pkg/tokenizer))
