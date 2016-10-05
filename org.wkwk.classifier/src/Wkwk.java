
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.jar.Attributes;
import org.wkwk.classifier.MyId3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Husni
 */
public class Wkwk {
    private DataSource dataSource;
    private Instances trainData;
    private Instances testData;
    private Classifier classifier;
    
    public void loadData(String filePath) throws Exception {
        dataSource = new DataSource(filePath);
        
        trainData = dataSource.getDataSet();
        trainData.setClassIndex(trainData.numAttributes() - 1);
        
        testData = dataSource.getDataSet();
        testData.setClassIndex(testData.numAttributes() - 1);
    }
    
    public void setClassifier(Classifier classifier) throws Exception {
        this.classifier = classifier;
        classifier.buildClassifier(trainData);
    }
    
    public void saveModel(String filePath) throws IOException, Exception {
        if (classifier == null) {
            throw new Exception();
        }
        
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
        oos.writeObject(classifier);
        oos.flush();
        oos.close();
    }
    
    public void removeAttribute() throws Exception {
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "1";
        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, remove);
    }
    
    public void loadModel(String filePath) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(
                           new FileInputStream(filePath));
        this.classifier = (Classifier) ois.readObject();
        ois.close();
    }
    
    public Evaluation evaluate() throws Exception {
        Evaluation eval = new Evaluation(testData);
        eval.crossValidateModel(classifier, testData, 10, new Random(1));
        
        return eval;
    }
}
