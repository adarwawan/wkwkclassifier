
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
import weka.filters.supervised.instance.Resample;
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
            throw new Exception("Classifier is not set");
        }
        
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
        oos.writeObject(classifier);
        oos.flush();
        oos.close();
    }
    
    public void removeAttribute(String options) throws Exception {        
        String[] optionsArr = weka.core.Utils.splitOptions(options);        
        Remove remove = new Remove();
        remove.setOptions(optionsArr);
        remove.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, remove);
    }
    
    public void resample(String options) throws Exception {
        String[] optionsArr = weka.core.Utils.splitOptions(options);
        Resample resample = new Resample();
        resample.setOptions(optionsArr);
        resample.setInputFormat(trainData);
        trainData = Filter.useFilter(trainData, resample);
    }
    
    public void loadModel(String filePath) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(
                           new FileInputStream(filePath));
        this.classifier = (Classifier) ois.readObject();
        ois.close();
    }
    
    public Evaluation evaluate(int evaluationID) throws Exception {
        Evaluation eval = new Evaluation(testData);
        switch(evaluationID) {
            case 1:
                eval.evaluateModel(classifier, testData);
                break;
            case 2:
                eval.crossValidateModel(classifier, testData, 10, new Random(1));
                break;
        }
        
        return eval;
    }
    
    public Evaluation evaluate(double percentage) throws Exception {
        System.out.println("WOW");
        int trainSize = (int) Math.round(trainData.numInstances() * percentage / 100);
        int testSize = trainData.numInstances() - trainSize;
        Instances train = new Instances(trainData, 0, trainSize);
        Instances test = new Instances(trainData, trainSize, testSize);
        
        classifier.buildClassifier(train);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(classifier, test);
        return eval;
    }
    
    public Instances getTrainData() {
        return trainData;
    }
}
