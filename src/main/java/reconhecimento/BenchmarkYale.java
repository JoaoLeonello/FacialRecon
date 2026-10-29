package reconhecimento;

import java.io.File;
import java.util.EnumMap;
import java.util.Map;
import reconhecimento.Reconhecedor.Algoritmo;
import reconhecimento.Reconhecedor.Amostras;

/**
 * Mede a taxa de acerto de cada algoritmo na base Yale Faces (15 pessoas).
 * Os arquivos se chamam subjectNN.condicao.png; NN é o rótulo.
 */
public final class BenchmarkYale {

    public static Map<Algoritmo, Double> executar(File base) {
        Amostras treino = Reconhecedor.carregar(new File(base, "treinamento"), BenchmarkYale::rotulo);
        Amostras teste = Reconhecedor.carregar(new File(base, "teste"), BenchmarkYale::rotulo);
        Reconhecedor reconhecedor = new Reconhecedor();

        Map<Algoritmo, Double> acertos = new EnumMap<>(Algoritmo.class);
        for (Algoritmo algoritmo : Algoritmo.values()) {
            reconhecedor.treinar(algoritmo, treino);
            int certos = 0;
            for (int i = 0; i < teste.faces().size(); i++) {
                if (reconhecedor.prever(teste.faces().get(i)).rotulo() == teste.rotulos()[i]) {
                    certos++;
                }
            }
            acertos.put(algoritmo, certos / (double) teste.faces().size());
        }
        return acertos;
    }

    static int rotulo(String arquivo) {
        return Integer.parseInt(arquivo.substring("subject".length(), "subjectNN".length()));
    }

    public static void main(String[] args) {
        File base = new File(args.length > 0 ? args[0] : "datasets/yalefaces");
        System.out.printf("Base: %s%n%n", base.getAbsolutePath());
        executar(base).forEach((algoritmo, taxa) ->
                System.out.printf("%-12s %5.1f%%%n", algoritmo, taxa * 100));
    }
}
