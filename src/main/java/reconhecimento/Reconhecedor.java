package reconhecimento;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToIntFunction;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

/** Detecção (Haar cascade) + reconhecimento (Eigenfaces, Fisherfaces ou LBPH) de rostos. */
public final class Reconhecedor {

    /** Todas as faces são normalizadas para este tamanho antes de treinar ou prever. */
    public static final Size TAMANHO_FACE = new Size(160, 160);

    public enum Algoritmo {
        // Limiares de distância padrão: acima disso a face é tratada como "Desconhecido".
        // Cada algoritmo mede distância numa escala diferente, então o valor muda por algoritmo.
        LBPH(85),
        EIGENFACES(4500),
        FISHERFACES(1500);

        public final double limiarPadrao;

        Algoritmo(double limiarPadrao) {
            this.limiarPadrao = limiarPadrao;
        }

        FaceRecognizer criar() {
            return switch (this) {
                case LBPH -> LBPHFaceRecognizer.create();
                case EIGENFACES -> EigenFaceRecognizer.create();
                case FISHERFACES -> FisherFaceRecognizer.create();
            };
        }
    }

    public record Predicao(int rotulo, double distancia) {}

    /** Amostras de treino: faces já normalizadas e o rótulo (id da pessoa) de cada uma. */
    public record Amostras(List<Mat> faces, int[] rotulos) {
        public long pessoasDistintas() {
            return Arrays.stream(rotulos).distinct().count();
        }
    }

    private final CascadeClassifier detector;
    private volatile FaceRecognizer modelo;

    public Reconhecedor() {
        try {
            // O CascadeClassifier só lê de arquivo; cacheResource extrai o XML do classpath (funciona dentro de jar).
            File cascade = Loader.cacheResource(Reconhecedor.class, "/haarcascade_frontalface_alt.xml");
            detector = new CascadeClassifier(cascade.getAbsolutePath());
        } catch (IOException e) {
            throw new UncheckedIOException("Não foi possível carregar o haarcascade", e);
        }
        if (detector.empty()) {
            throw new IllegalStateException("haarcascade_frontalface_alt.xml inválido");
        }
    }

    public static Mat paraCinza(Mat colorida) {
        Mat cinza = new Mat();
        cvtColor(colorida, cinza, COLOR_BGR2GRAY);
        return cinza;
    }

    /** Recorta, redimensiona e equaliza o histograma: reduz o efeito de iluminação na comparação. */
    public static Mat normalizar(Mat cinza) {
        Mat face = new Mat();
        resize(cinza, face, TAMANHO_FACE);
        equalizeHist(face, face);
        return face;
    }

    public List<Rect> detectar(Mat cinza) {
        RectVector faces = new RectVector();
        detector.detectMultiScale(cinza, faces, 1.1, 4, 0, new Size(80, 80), new Size());
        List<Rect> lista = new ArrayList<>();
        for (long i = 0; i < faces.size(); i++) {
            lista.add(new Rect(faces.get(i)));
        }
        return lista;
    }

    public void treinar(Algoritmo algoritmo, Amostras amostras) {
        if (amostras.faces().isEmpty()) {
            throw new IllegalArgumentException("Nenhuma face cadastrada para treinar");
        }
        if (algoritmo == Algoritmo.FISHERFACES && amostras.pessoasDistintas() < 2) {
            throw new IllegalArgumentException("Fisherfaces precisa de pelo menos 2 pessoas cadastradas");
        }
        MatVector fotos = new MatVector(amostras.faces().toArray(Mat[]::new));
        Mat rotulos = new Mat(amostras.rotulos().length, 1, CV_32SC1);
        IntBuffer buffer = rotulos.createBuffer();
        buffer.put(amostras.rotulos());

        FaceRecognizer novo = algoritmo.criar();
        // O treino do Fisherfaces (LAPACK via OpenBLAS) estoura a pilha padrão e derruba a JVM com SIGSEGV;
        // por isso o treino roda numa thread com pilha grande.
        RuntimeException[] erro = new RuntimeException[1];
        Thread treino = new Thread(null, () -> {
            try {
                novo.train(fotos, rotulos);
            } catch (RuntimeException e) {
                erro[0] = e;
            }
        }, "treino-" + algoritmo, 256L * 1024 * 1024);
        treino.start();
        try {
            treino.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Treino interrompido", e);
        }
        if (erro[0] != null) {
            throw erro[0];
        }
        modelo = novo;
    }

    public boolean treinado() {
        return modelo != null;
    }

    /** @param face face já normalizada com {@link #normalizar(Mat)} */
    public Predicao prever(Mat face) {
        FaceRecognizer atual = modelo;
        if (atual == null) {
            throw new IllegalStateException("Modelo ainda não treinado");
        }
        IntPointer rotulo = new IntPointer(1);
        DoublePointer distancia = new DoublePointer(1);
        atual.predict(face, rotulo, distancia);
        return new Predicao(rotulo.get(0), distancia.get(0));
    }

    /** Lê todas as imagens de um diretório, extraindo o rótulo do nome do arquivo. */
    public static Amostras carregar(File diretorio, ToIntFunction<String> rotuloDoNome) {
        File[] arquivos = diretorio.listFiles((dir, nome) -> nome.matches("(?i).*\\.(png|jpe?g|gif|bmp)$"));
        if (arquivos == null) {
            return new Amostras(List.of(), new int[0]);
        }
        Arrays.sort(arquivos);
        List<Mat> faces = new ArrayList<>();
        int[] rotulos = new int[arquivos.length];
        for (File arquivo : arquivos) {
            Mat foto = imread(arquivo.getAbsolutePath(), IMREAD_GRAYSCALE);
            if (foto.empty()) {
                throw new IllegalStateException("Imagem ilegível: " + arquivo);
            }
            rotulos[faces.size()] = rotuloDoNome.applyAsInt(arquivo.getName());
            faces.add(normalizar(foto));
        }
        return new Amostras(faces, rotulos);
    }
}
