package reconhecimento;

import java.io.File;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_face.createEigenFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

public class TreinamentoYale {
    public static void main(String[] args) {
        File diretorio = new File("src\\yalefaces\\treinamento");
        File[] arquivos = diretorio.listFiles();
        MatVector fotos = new MatVector(arquivos.length);
        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
      
        for (File imagem : arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().substring(7,9));
            resize(foto, foto, new Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }
        
        FaceRecognizer eigenface = createEigenFaceRecognizer(30, 0);
        FaceRecognizer fisherface = createFisherFaceRecognizer(30, 0);        
        FaceRecognizer lbph = createLBPHFaceRecognizer(12, 10, 15, 15, 0);

        eigenface.train(fotos, rotulos);
        eigenface.save("src\\recursos\\classificadorEigenfacesYale.yml");
        
        fisherface.train(fotos, rotulos);
        fisherface.save("src\\recursos\\classificadorFisherfacesYale.yml");
        
        lbph.train(fotos, rotulos);
        lbph.save("src\\recursos\\classificadorLBPHYale.yml");
    }
}