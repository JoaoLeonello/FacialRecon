package reconhecimento;

import static org.bytedeco.opencv.global.opencv_core.flip;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.swing.BorderFactory;
import javax.swing.DefaultListModel;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import reconhecimento.Reconhecedor.Algoritmo;
import reconhecimento.Reconhecedor.Predicao;

/** Interface gráfica: câmera ao vivo, cadastro de pessoas e reconhecimento em tempo real. */
public final class App {

    private static final int AMOSTRAS_POR_CADASTRO = 30;
    // Guarda 1 a cada N quadros no cadastro, para as amostras variarem um pouco entre si.
    private static final int INTERVALO_CAPTURA = 3;

    private static final Color VERDE = new Color(80, 200, 80);
    private static final Color VERMELHO = new Color(230, 60, 60);
    private static final Color AMARELO = new Color(250, 210, 0);
    private static final Color CINZA = new Color(180, 180, 180);

    /** Rótulo desenhado sobre um rosto (em Java2D: o putText do OpenCV não desenha acentos). */
    private record Marcacao(Rect face, String texto, Color cor) {}

    private final int indiceCamera;
    private final BaseDeRostos base = new BaseDeRostos(Path.of("dados"));
    private final Reconhecedor reconhecedor = new Reconhecedor();
    private final ExecutorService fila = Executors.newSingleThreadExecutor();

    private final JFrame janela = new JFrame("FacialRecon");
    private final JLabel video = new JLabel("Abrindo câmera...", SwingConstants.CENTER);
    private final JLabel status = new JLabel(" ");
    private final JTextField campoNome = new JTextField(14);
    private final JButton botaoCadastrar = new JButton("Cadastrar rosto");
    private final JComboBox<Algoritmo> comboAlgoritmo = new JComboBox<>(Algoritmo.values());
    private final JSpinner spinnerLimiar = new JSpinner(
            new SpinnerNumberModel(Algoritmo.LBPH.limiarPadrao, 0, 100_000, 5));
    private final DefaultListModel<String> listaPessoas = new DefaultListModel<>();

    private volatile boolean rodando = true;
    private volatile int idCadastro;
    private volatile int faltamCapturar;

    private App(int indiceCamera) {
        this.indiceCamera = indiceCamera;
    }

    public static void main(String[] args) {
        int camera = args.length > 0 ? Integer.parseInt(args[0]) : 0;
        SwingUtilities.invokeLater(() -> new App(camera).abrir());
    }

    private void abrir() {
        video.setPreferredSize(new Dimension(640, 480));
        video.setOpaque(true);
        video.setBackground(Color.BLACK);
        video.setForeground(Color.LIGHT_GRAY);

        botaoCadastrar.addActionListener(e -> iniciarCadastro());
        campoNome.addActionListener(e -> iniciarCadastro());
        comboAlgoritmo.addActionListener(e -> {
            Algoritmo a = (Algoritmo) comboAlgoritmo.getSelectedItem();
            spinnerLimiar.setValue(a.limiarPadrao);
            retreinar();
        });

        JPanel controles = new JPanel(new GridBagLayout());
        controles.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(3, 0, 3, 0);
        controles.add(new JLabel("Nome da pessoa"), c);
        controles.add(campoNome, c);
        controles.add(botaoCadastrar, c);
        c.insets = new Insets(15, 0, 3, 0);
        controles.add(new JLabel("Algoritmo"), c);
        c.insets = new Insets(3, 0, 3, 0);
        controles.add(comboAlgoritmo, c);
        controles.add(new JLabel("Limiar de distância"), c);
        spinnerLimiar.setToolTipText("Acima deste valor a face é marcada como Desconhecido");
        controles.add(spinnerLimiar, c);
        c.insets = new Insets(15, 0, 3, 0);
        controles.add(new JLabel("Pessoas cadastradas"), c);
        c.insets = new Insets(3, 0, 3, 0);
        c.fill = GridBagConstraints.BOTH;
        c.weighty = 1;
        controles.add(new JScrollPane(new JList<>(listaPessoas)), c);

        status.setBorder(BorderFactory.createEmptyBorder(4, 10, 4, 10));

        janela.setLayout(new BorderLayout());
        janela.add(video, BorderLayout.CENTER);
        janela.add(controles, BorderLayout.EAST);
        janela.add(status, BorderLayout.SOUTH);
        janela.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        janela.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                rodando = false;
                fila.shutdownNow();
            }
        });
        janela.pack();
        janela.setLocationRelativeTo(null);
        janela.setVisible(true);

        atualizarListaPessoas();
        retreinar();
        Thread loop = new Thread(this::loopCamera, "camera");
        loop.setDaemon(true);
        loop.start();
    }

    private void iniciarCadastro() {
        String nome = campoNome.getText().trim();
        if (nome.isEmpty()) {
            mostrarStatus("Digite um nome antes de cadastrar.");
            return;
        }
        idCadastro = base.idPara(nome);
        faltamCapturar = AMOSTRAS_POR_CADASTRO;
        botaoCadastrar.setEnabled(false);
        atualizarListaPessoas();
        mostrarStatus("Olhe para a câmera e mexa um pouco a cabeça. Capturando " + nome + "...");
    }

    private void retreinar() {
        Algoritmo algoritmo = (Algoritmo) comboAlgoritmo.getSelectedItem();
        fila.submit(() -> {
            Reconhecedor.Amostras amostras = base.amostras();
            if (amostras.faces().isEmpty()) {
                mostrarStatus("Nenhuma pessoa cadastrada. Digite um nome e clique em \"Cadastrar rosto\".");
                return;
            }
            mostrarStatus("Treinando " + algoritmo + "...");
            try {
                reconhecedor.treinar(algoritmo, amostras);
                mostrarStatus("Modelo " + algoritmo + " pronto: " + amostras.faces().size() + " fotos de "
                        + amostras.pessoasDistintas() + " pessoa(s).");
            } catch (RuntimeException e) {
                mostrarStatus("Não foi possível treinar: " + e.getMessage());
            }
        });
    }

    private void loopCamera() {
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(indiceCamera);
        OpenCVFrameConverter.ToMat paraMat = new OpenCVFrameConverter.ToMat();
        Java2DFrameConverter paraImagem = new Java2DFrameConverter();
        try {
            camera.start();
        } catch (Exception e) {
            SwingUtilities.invokeLater(() -> JOptionPane.showMessageDialog(janela,
                    "Não foi possível abrir a câmera " + indiceCamera + ":\n" + e.getMessage(),
                    "FacialRecon", JOptionPane.ERROR_MESSAGE));
            return;
        }
        try {
            long quadro = 0;
            while (rodando) {
                Frame frame = camera.grab();
                if (frame == null) {
                    break;
                }
                Mat imagem = paraMat.convert(frame);
                flip(imagem, imagem, 1); // espelha, como um espelho/selfie
                List<Marcacao> marcacoes = processar(imagem, quadro++);
                BufferedImage bi = Java2DFrameConverter.cloneBufferedImage(paraImagem.convert(paraMat.convert(imagem)));
                desenhar(bi, marcacoes);
                SwingUtilities.invokeLater(() -> {
                    video.setText(null);
                    video.setIcon(new ImageIcon(bi));
                });
            }
        } catch (Exception e) {
            mostrarStatus("Erro na câmera: " + e.getMessage());
        } finally {
            try {
                camera.stop();
                camera.release();
            } catch (Exception ignored) {
                // fechando o programa
            }
        }
    }

    private List<Marcacao> processar(Mat imagem, long quadro) {
        Mat cinza = Reconhecedor.paraCinza(imagem);
        List<Rect> faces = reconhecedor.detectar(cinza);
        List<Marcacao> marcacoes = new ArrayList<>();

        if (faltamCapturar > 0) {
            for (Rect face : faces) {
                marcacoes.add(new Marcacao(face, "Capturando " + (AMOSTRAS_POR_CADASTRO - faltamCapturar) + "/"
                        + AMOSTRAS_POR_CADASTRO, AMARELO));
            }
            // Só captura com exatamente um rosto na tela, para não cadastrar a pessoa errada.
            if (faces.size() == 1 && quadro % INTERVALO_CAPTURA == 0) {
                base.salvarFace(idCadastro, Reconhecedor.normalizar(new Mat(cinza, faces.get(0))));
                if (--faltamCapturar == 0) {
                    SwingUtilities.invokeLater(() -> {
                        botaoCadastrar.setEnabled(true);
                        campoNome.setText("");
                    });
                    retreinar();
                }
            }
            return marcacoes;
        }

        double limiar = ((Number) spinnerLimiar.getValue()).doubleValue();
        for (Rect face : faces) {
            if (!reconhecedor.treinado()) {
                marcacoes.add(new Marcacao(face, "?", CINZA));
                continue;
            }
            Predicao p = reconhecedor.prever(Reconhecedor.normalizar(new Mat(cinza, face)));
            boolean conhecido = p.distancia() <= limiar;
            String texto = (conhecido ? base.nome(p.rotulo()) : "Desconhecido")
                    + String.format(" (%.0f)", p.distancia());
            marcacoes.add(new Marcacao(face, texto, conhecido ? VERDE : VERMELHO));
        }
        return marcacoes;
    }

    private static void desenhar(BufferedImage imagem, List<Marcacao> marcacoes) {
        Graphics2D g = imagem.createGraphics();
        g.setStroke(new BasicStroke(2));
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18));
        for (Marcacao m : marcacoes) {
            Rect r = m.face();
            g.setColor(m.cor());
            g.drawRect(r.x(), r.y(), r.width(), r.height());
            g.drawString(m.texto(), r.x(), Math.max(r.y() - 8, 18));
        }
        g.dispose();
    }

    private void atualizarListaPessoas() {
        listaPessoas.clear();
        base.pessoas().values().stream().sorted(String.CASE_INSENSITIVE_ORDER).forEach(listaPessoas::addElement);
    }

    private void mostrarStatus(String texto) {
        SwingUtilities.invokeLater(() -> status.setText(texto));
    }
}
