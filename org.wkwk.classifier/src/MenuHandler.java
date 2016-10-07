
import java.util.Scanner;
import org.wkwk.classifier.MyC45;
import org.wkwk.classifier.MyId3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author husni
 */
public class MenuHandler {
    Wkwk wkwk;
    Scanner scanner;
    
    public MenuHandler() {
        wkwk = new Wkwk();
        scanner = new Scanner(System.in);
    }
    
    public void loadDataMenu() throws Exception {
        String filePath;
        System.out.println("Dataset file path: ");
        filePath = scanner.nextLine();
        wkwk.loadData(filePath);
        
        System.out.println("Data has ben loaded succesfully");
    }
    
    public void removeAttributeMenu() throws Exception {
        Instances trainData = wkwk.getTrainData();
        for (int i = 0 ; i < trainData.numAttributes(); i++) {
            System.out.println(trainData.attribute(i));
        }
        
        System.out.println("Type cancel to cancel");
        System.out.println("Remove parameter: ");
        String options = scanner.nextLine();
        if (!options.equals("cancel")) {
            wkwk.removeAttribute(options);
        }
    }
    
    public void buildClassifierMenu() throws Exception {
        System.out.println("==========================");
        System.out.println("==== Pilih Classifier ====");
        System.out.println("==========================");
        
        System.out.println("1. Naive Bayes");
        System.out.println("2. MyId3");
        System.out.println("3. MyC45");
        
        
        int classifierID = scanner.nextInt();
        Classifier classifier = null;
        switch(classifierID) {
            case 1:
                classifier = new NaiveBayes();
                break;
            case 2:
                classifier = new MyId3();
                break;
            case 3:
                classifier = new MyC45();
        }
        
        wkwk.setClassifier(classifier);
    }
    
    public void testModelMenu() throws Exception {
        System.out.println("==========================");
        System.out.println("===== Pilih Test Set =====");
        System.out.println("==========================");
        
        System.out.println("1. Full train");
        System.out.println("2. 10 folds cross validation");
        System.out.println("3. Percentage Split");

        int evaluationID = scanner.nextInt();

        
        Evaluation eval = null;
        if (evaluationID == 3) {
            System.out.println("Persentase: ");
            double percentage = scanner.nextDouble();
            eval = wkwk.evaluate(percentage);
        } else {
            eval = wkwk.evaluate(evaluationID);
        }
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        
        System.out.println("\n=== Matriks ===");
        System.out.println(eval.toMatrixString());
        
        
    }
    
    public void saveModelMenu() throws Exception {
        System.out.println("File location: ");
        String filePath = scanner.nextLine();
        wkwk.saveModel(filePath);
    }
    
    public void loadModelMenu() throws Exception {
        System.out.println("Model location: ");
        String filePath = scanner.nextLine();
        wkwk.loadModel(filePath);
    }
}
