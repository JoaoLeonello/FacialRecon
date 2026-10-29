# FacialRecon

Reconhecimento facial em tempo real feito em Java com [JavaCV](https://github.com/bytedeco/javacv), o wrapper Java do OpenCV. Você abre o app, cadastra o rosto de algumas pessoas pela webcam e ele passa a dizer quem está na frente da câmera.

## Como funciona

1. A câmera captura um quadro e o converte para tons de cinza.
2. Um classificador Haar cascade (`haarcascade_frontalface_alt.xml`) encontra os rostos no quadro.
3. Cada rosto é recortado, redimensionado para 160x160 e tem o histograma equalizado, o que reduz o efeito da iluminação.
4. Um dos três algoritmos clássicos do módulo `face` do OpenCV compara esse rosto com as fotos cadastradas e devolve a pessoa mais parecida e uma distância.
5. Se a distância passar do limiar configurado, o rosto aparece como "Desconhecido".

Os três algoritmos disponíveis:

| Algoritmo | Ideia | Acerto na Yale Faces |
|---|---|---|
| Eigenfaces | PCA: projeta os rostos nas direções de maior variação | 86,7% |
| Fisherfaces | LDA: procura as direções que mais separam uma pessoa da outra. Precisa de pelo menos 2 pessoas | 100% |
| LBPH | Histogramas de padrões binários locais, por região do rosto. Lida melhor com iluminação e é o padrão do app | 83,3% |

Os números vêm do benchmark incluído no projeto (`BenchmarkYale`), rodado sobre 133 fotos de treino e 30 de teste de 15 pessoas.

## Requisitos

- Java 17 ou mais recente
- Maven 3.8 ou mais recente
- Uma webcam

As bibliotecas nativas do OpenCV vêm pelo Maven, então não é preciso instalar o OpenCV no sistema.

## Rodando

```bash
mvn compile exec:java
```

O primeiro build baixa os binários do OpenCV para todos os sistemas operacionais. Para baixar só os do seu, informe a plataforma:

```bash
mvn -Djavacpp.platform=linux-x86_64 compile exec:java    # ou windows-x86_64, macosx-arm64, macosx-x86_64
```

Se tiver mais de uma câmera, passe o índice dela:

```bash
mvn compile exec:java -Dexec.args="1"
```

### Usando o app

1. Digite um nome e clique em "Cadastrar rosto". Olhe para a câmera e mexa um pouco a cabeça enquanto ele captura 30 fotos. A captura só acontece quando há um único rosto na imagem.
2. Ao terminar, o modelo é treinado de novo sozinho e o reconhecimento começa: verde para pessoa conhecida, vermelho para desconhecida. O número entre parênteses é a distância.
3. Troque o algoritmo no seletor para comparar. Ajuste o limiar se o app confundir pessoas (diminua) ou não reconhecer ninguém (aumente).

As fotos e os nomes ficam em `dados/`, fora do controle de versão. Para remover uma pessoa, apague as fotos `dados/faces/pessoa.ID.*.png` e a linha correspondente em `dados/pessoas.properties`.

## Benchmark e testes

```bash
mvn exec:java -Dexec.mainClass=reconhecimento.BenchmarkYale   # imprime a taxa de acerto de cada algoritmo
mvn test                                                      # falha se algum algoritmo ficar abaixo de 60%
```

## Estrutura

```
src/main/java/reconhecimento/
  App.java            interface Swing: câmera, cadastro e reconhecimento
  Reconhecedor.java   detecção de rostos, normalização, treino e predição
  BaseDeRostos.java   pessoas cadastradas e suas fotos em disco
  BenchmarkYale.java  avaliação dos algoritmos na base Yale Faces
src/main/resources/   haarcascade_frontalface_alt.xml
datasets/yalefaces/   base Yale Faces separada em treinamento/ e teste/
```

## Limitações

Eigenfaces, Fisherfaces e LBPH são técnicas dos anos 90 e 2000. Funcionam bem em ambiente controlado, com poucas pessoas e boa luz, mas erram bem mais que redes neurais modernas como FaceNet ou ArcFace quando mudam ângulo, iluminação ou acessórios. O projeto tem fins de estudo e não deve ser usado como controle de acesso.

## Créditos

A base [Yale Faces](http://vision.ucsd.edu/content/yale-face-database) é do Yale Center for Computational Vision and Control. Projeto feito por diversão e para aprender.
