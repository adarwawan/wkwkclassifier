package org.wkwk.classifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 *
 * @author Husni
 */
public class MyId3 extends AbstractClassifier {
    
    public MyId3[] successors;
    public Attribute classAttribute;
    public Attribute attribute;
    public double classValue;
    
    /**
     * 
     * @param data
     */
    @Override
    public void buildClassifier(Instances data) {
        data.deleteWithMissingClass();
        makeTree(data);
    }
    
    /**
     * Build the decision tree with Id3 Algorithm
     * @param data Dataset
     */
    public void makeTree(Instances data) {
        // Calculate information gain for all attributes, except class attribute
        double[] infoGains = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            infoGains[i] = computeInfoGain(data, data.attribute(i));
        }
        // Find attribute with highest information gain
        attribute = data.attribute(Utils.maxIndex(infoGains));
        
        // If infogain is 0, create a leaf node with class value = most common value in class attribute
        double[] classCount;
        if (Utils.eq(infoGains[attribute.index()], 0)) {
            attribute = null;
            successors = null;
            classCount = new double[data.numClasses()];
            
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                classCount[(int) inst.classValue()]++;
            }
            
            classValue = Utils.maxIndex(classCount);
            classAttribute = data.classAttribute();
        } else {
            Instances[] splitData = splitData(data, attribute);
            successors = new MyId3[attribute.numValues()];
            for (int i = 0; i < attribute.numValues(); i++) {
                successors[i] = new MyId3();
                successors[i].makeTree(splitData[i]);
            }
        }
    }   
    
    /**
     * Compute information gain of the attribute
     * @param data Dataset
     * @param attr Attribute
     * @return Information gain
     */
    public double computeInfoGain(Instances data, Attribute attr) {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, attr);
        
        for (int i = 0; i < attr.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= ((double) splitData[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }
    
    /**
     * Compute entropy of dataset
     * @param data 
     * @return Entropy of given data.
     */
    public double computeEntropy(Instances data) {
        // Hitung kemunculan kelas
        double[] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        
        // Hitung entropy
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (classCounts[i] > 0) {
                entropy -= classCounts[i]/data.numInstances()* Utils.log2(classCounts[i]/data.numInstances());
            }
        }
        return entropy;
    }
    
    /**
     * Split dataset based according to attr
     * @param data Dataset that will be split
     * @param attr Attribute used for splitting
     * @return Split dataset
     */
    public Instances[] splitData(Instances data, Attribute attr) {
        Instances[] splitData = new Instances[attr.numValues()];
        for (int i = 0; i < attr.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        } 
        
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(attr)].add(inst);
        }
        
        return splitData;
    }
    
    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the class value
     * @throws NoSupportForMissingValueException
     */
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("ID3 Algorithm doesn't support missing values");
        }
        // If it's leaf node, then return the class value.
        if (attribute == null) {
            return classValue;
        } else {
            return successors[(int) instance.value(attribute)].classifyInstance(instance);
        }
    }
}
