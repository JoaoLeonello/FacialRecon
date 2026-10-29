package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Pessoas cadastradas e suas fotos, em disco:
 * <pre>
 * dados/pessoas.properties   id=nome
 * dados/faces/pessoa.ID.N.png
 * </pre>
 */
public final class BaseDeRostos {

    private final Path faces;
    private final Path arquivoPessoas;
    private final TreeMap<Integer, String> pessoas = new TreeMap<>();

    public BaseDeRostos(Path raiz) {
        this.faces = raiz.resolve("faces");
        this.arquivoPessoas = raiz.resolve("pessoas.properties");
        try {
            Files.createDirectories(faces);
            if (Files.exists(arquivoPessoas)) {
                Properties props = new Properties();
                try (Reader r = Files.newBufferedReader(arquivoPessoas, StandardCharsets.UTF_8)) {
                    props.load(r);
                }
                props.forEach((id, nome) -> pessoas.put(Integer.parseInt((String) id), (String) nome));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public synchronized Map<Integer, String> pessoas() {
        return Map.copyOf(pessoas);
    }

    public synchronized String nome(int id) {
        return pessoas.getOrDefault(id, "Desconhecido");
    }

    /** Retorna o id da pessoa com esse nome, criando o cadastro se ainda não existir. */
    public synchronized int idPara(String nome) {
        for (Map.Entry<Integer, String> e : pessoas.entrySet()) {
            if (e.getValue().equalsIgnoreCase(nome)) {
                return e.getKey();
            }
        }
        int id = pessoas.isEmpty() ? 1 : pessoas.lastKey() + 1;
        pessoas.put(id, nome);
        salvarPessoas();
        return id;
    }

    public synchronized void salvarFace(int id, Mat face) {
        int n = 1;
        while (Files.exists(caminho(id, n))) {
            n++;
        }
        if (!imwrite(caminho(id, n).toString(), face)) {
            throw new IllegalStateException("Falha ao gravar " + caminho(id, n));
        }
    }

    public Reconhecedor.Amostras amostras() {
        return Reconhecedor.carregar(faces.toFile(), nome -> Integer.parseInt(nome.split("\\.")[1]));
    }

    private Path caminho(int id, int n) {
        return faces.resolve("pessoa." + id + "." + n + ".png");
    }

    private void salvarPessoas() {
        Properties props = new Properties();
        pessoas.forEach((id, nome) -> props.setProperty(id.toString(), nome));
        try (Writer w = Files.newBufferedWriter(arquivoPessoas, StandardCharsets.UTF_8)) {
            props.store(w, "FacialRecon - pessoas cadastradas");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
