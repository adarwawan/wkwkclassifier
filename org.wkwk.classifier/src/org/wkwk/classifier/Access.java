/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wkwk.classifier;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.*;
import weka.filters.Filter;
import weka.filters.supervised.instance.*;
import weka.filters.unsupervised.attribute.*;


/**
 *
 * @author adarwawan
 */
public class Access {

    /**
     * @param args the command line arguments
     * args[0] = filename train set
     * args[1] = filename test set
     * args[2] = remove attribute
     * args[3] = bias resample
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Masukan salah...");
            System.exit(1);
        }
        
        
        // Read Dataset (arff, csv)
        DataSource source = new DataSource(args[0]);
        DataSource testSource = new DataSource(args[1]);
        Instances data = source.getDataSet();
        
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        // Remove attr
        String[] rangeOps = new String[2];
        rangeOps[0] = "-R";                                    // "range"
        rangeOps[1] = args[2];                                 // first attribute
        Remove remove = new Remove();                         // new instance of filter
        remove.setOptions(rangeOps);                           // set options
        remove.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
        Instances newData = Filter.useFilter(data, remove);   // apply filter
        
        // Filter Resample
        String[] biasOps = new String[2];
        biasOps[0] = "-B";                                    // "range"
        biasOps[1] = args[3];                                 // first attribute
        Resample resample = new Resample();
        resample.setOptions(biasOps);
        resample.setInputFormat(data);
        newData = Filter.useFilter(data, resample);
        
        // Build Classifier
        MyId3 tree = new MyId3();         // new instance of tree
        tree.buildClassifier(data);   // build classifier
        
        // Evaluation with test set
        Instances testSet = testSource.getDataSet();
        // train classifier
        Classifier cls = new MyId3();
        cls.buildClassifier(data);
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(cls, testSet);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        // Evaluation with 10 Fold-CV
        Evaluation evalCV = new Evaluation(newData);
        evalCV.crossValidateModel(tree, newData, 10, new Random(1));
    }
    
}
