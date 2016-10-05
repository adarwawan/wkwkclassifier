
import java.util.Scanner;
import org.wkwk.classifier.MyId3;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;

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
        System.out.println("Remove parameter: ");
        String options = scanner.nextLine();
        wkwk.removeAttribute();
    }
    
    public void buildClassifierMenu() throws Exception {
        System.out.println("==========================");
        System.out.println("==== Pilih Classifier ====");
        System.out.println("==========================");
        
        System.out.println("1. Naive Bayes");
        System.out.println("2. MyId3");
        System.out.println("3. J48");
        
        
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
                classifier = new J48();
        }
        
        wkwk.setClassifier(classifier);
    }
    
    public void testModelMenu() throws Exception {
        Evaluation eval = wkwk.evaluate();
        System.out.println(eval.toSummaryString());
    }
}