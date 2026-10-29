package reconhecimento;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.Map;
import org.junit.jupiter.api.Test;
import reconhecimento.Reconhecedor.Algoritmo;

class BenchmarkYaleTest {

    @Test
    void rotuloVemDoNomeDoArquivo() {
        assertEquals(7, BenchmarkYale.rotulo("subject07.sad.png"));
        assertEquals(15, BenchmarkYale.rotulo("subject15.wink.png"));
    }

    @Test
    void todosOsAlgoritmosReconhecemBemAYale() {
        Map<Algoritmo, Double> acertos = BenchmarkYale.executar(new File("datasets/yalefaces"));
        acertos.forEach((algoritmo, taxa) ->
                assertTrue(taxa >= 0.6, algoritmo + " acertou só " + taxa));
    }
}
