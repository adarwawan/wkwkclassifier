package org.wkwk.classifier;

import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author adarwawan
 */
public class MyC45 extends AbstractClassifier {

    /**
     * The node's successors 
     */
    private MyC45[] successors;
    
    /**
     * Attribute for splitting
     */
    private Attribute splitAttribute;
    
    /**
     * Class value if node is leaf.
     */
    private double classValue;
    
    /**
     * Class distribution if node is leaf.
     */
    private double[] classDistribution;

    /**
     * True if the tree is to be pruned.
     */
    private boolean isPruned = true;
    
    /**
     * Class attribute of dataset.
     */
    private Attribute classAttribute;

    /**
     * Threshold for numeric value.
     */
    private double attrThreshold;

        
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            for (int j = 0; j < 10; j++) {
                Instance instance = data.instance(j);
                if (instance.isMissing(attr)) {
                    instance.setValue(attr, fillMissingValue(data, attr));
                }
            }
        }
        
        data.deleteWithMissingClass();
        makeTree(data);
    }

    public double fillMissingValue(Instances data, Attribute attr) {
        int[] sum = new int[attr.numValues()];
        for (int i = 0; i < data.numInstances(); ++i) {
            sum[(int)data.instance(i).value(attr)]++;
        }
        return sum[Utils.maxIndex(sum)];
    }
    
    @Override
    public double classifyInstance(Instance data) {
        if (splitAttribute == null) {
            return classValue;
        }
        else {
            if (splitAttribute.isNominal()) {
                return successors[(int) data.value(splitAttribute)].classifyInstance(data);
            }
            else if (splitAttribute.isNumeric()) {
                if (data.value(splitAttribute) < attrThreshold) {
                    return successors[0].classifyInstance(data);
                }
                else {
                    return successors[1].classifyInstance(data);                    
                }
            }
            else {
                return -1;
            }
        }
    }
    
    public void makeTree(Instances data) {
        if (isPruned) {
            data = prune(data);
        }
        
        if (data.numInstances() == 0) {
            splitAttribute = null;  
        }
        
        // Calculate information gain for all attributes, except class attribute
        double[] infoGains = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            Attribute m_attr = data.attribute(i);
            if (m_attr.isNominal()) {
                infoGains[i] = computeInfoGain(data, data.attribute(i));                
            }
            else if (m_attr.isNumeric()) {
                infoGains[i] = computeInfoGainCont(data, data.attribute(i), bestThreshold(data, m_attr));
            }
        }
        splitAttribute = data.attribute(Utils.maxIndex(infoGains));
        if (splitAttribute.isNumeric()) {
            attrThreshold = bestThreshold(data, splitAttribute);
        }
        
        if (Utils.eq(infoGains[splitAttribute.index()], 0)) {
            splitAttribute = null;
            classDistribution = new double[data.numClasses()];
            for (int i = 0; i < data.numInstances(); i++) {
                int inst = (int) data.instance(i).value(data.classAttribute());
                classDistribution[inst]++;
            }
            Utils.normalize(classDistribution);
            classValue = Utils.maxIndex(classDistribution);
            classAttribute = data.classAttribute();
        }
        else {
            Instances[] splitData = null;
            if (splitAttribute.isNominal()) {
                splitData = splitData(data, splitAttribute);
            } 
            else if (splitAttribute.isNumeric()) {
                splitData = splitDataCont(data, splitAttribute, attrThreshold);
            }
            
            if (splitAttribute.isNominal()) {
                successors = new MyC45[splitAttribute.numValues()];
                for (int i = 0; i < splitAttribute.numValues(); i++) {
                    successors[i] = new MyC45();
                    successors[i].makeTree(splitData[i]);
                }
            }
            else if (splitAttribute.isNumeric()) {
                successors = new MyC45[2];
                for (int i = 0; i < 2; i++) {
                    successors[i] = new MyC45();
                    successors[i].makeTree(splitData[i]);
                }
            }
        }
    }

    // Implementasi
    public Instances prune(Instances data) {
        return data;
    }

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
    
    public double computeInfoGainCont(Instances data, Attribute attr, double threshold) {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitDataCont(data, attr, threshold);
        for (int i = 0; i < 2; i++) {
            if (splitData[i].numInstances() > 0) {
                infoGain -= ((double) splitData[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[i]);
            }
        }
        return infoGain;
    }

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
    
    public Instances[] splitDataCont(Instances data, Attribute attr, double threshold) {
        Instances[] splitData = new Instances[2];
        for (int i = 0; i < 2; i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            double temp = data.instance(i).value(attr);
            if (temp < threshold) {
                splitData[0].add(data.instance(i));
            } else {
                splitData[1].add(data.instance(i));
            }
        }
        return splitData;
    }

    public double bestThreshold(Instances data, Attribute attr) {
        data.sort(attr);
        
        double m_ig = 0;
        double bestThr = 0;
        double classTemp = data.get(0).classValue();
        double valueTemp = data.get(0).value(attr);
        
        Enumeration instEnum = data.enumerateInstances();
        double dt;
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if (classTemp != inst.classValue()) {
                classTemp = inst.classValue();
                dt = valueTemp;
                valueTemp = inst.value(attr);
                double threshold = dt + ((valueTemp - dt) / 2);
                double igTemp = computeInfoGainCont(data, attr, threshold);
                if (m_ig < igTemp) {
                    m_ig = igTemp;
                    bestThr = threshold;
                }
            }
        }
        return bestThr;
    }
}
