/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wkwk.classifier;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;


/**
 *
 * @author Husni
 */
public class Id3 
        extends Classifier
        implements TechnicalInformationHandler, Sourcable {
    
    private Id3[] m_Successors;
    private Attribute m_Attribute;
    private double m_ClassValue;
    private double[] m_Distribution;
    private Attribute m_ClassAttribute;
    
    
    public Capabilities getCapabilites() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.MISSING_CLASS_VALUES);
        
        result.setMinimumNumberInstances(0);
        
        return result;
    }
    
    private void makeTree(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_ClassValue = Instance.missingValue();
            m_Distribution = new double[data.numClasses()];
            
            return;
        }
        
        // Compute attribute with maximum information gain
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            infoGains[att.index()] = computeInfoGain(data, att);
        }
        m_Attribute = data.attribute(Utils.maxIndex(infoGains));
        
        // Make leaf if information gain is zero
        // otherwise create successor
        if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
            m_Attribute = null;
            m_Distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                m_Distribution[(int) inst.classValue()]++;
            }
            
            Utils.normalize(m_Distribution);
            m_ClassValue = Utils.maxIndex(m_Distribution);
            m_ClassAttribute = data.classAttribute();
        } else {
            Instances[] splitData = splitData(data, m_Attribute);
            m_Successors = new Id3[m_Attribute.numValues()];
            for (int i = 0; i < m_Attribute.numValues(); i++) {
                m_Successors[i] = new Id3();
                m_Successors[i].makeTree(splitData[i]);
            }
        }
    }
    
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int i = 0; i < att.numValues(); i++) {
            infoGain -= ((double) splitData[i].numInstances() / (double) data.numInstances()) * computeEntropy(splitData[i]);
        }
        
        return infoGain;
    }
    
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];
        
        // Enumerasi semua instance untuk menghitung jumlah kemunculan kelas
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        
        // Hitung entropy
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (classCounts[i] > 0) {
                entropy -= classCounts[i]/data.numInstances() * Utils.log2(classCounts[i]/data.numInstances());
            }
        }
        
        return entropy;
    }
    
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }
        
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);   
        
        // remove instance with missing class;
        data = new Instances(data);
        data.deleteWithMissingClass();
        
        makeTree(data);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
    
        result = new TechnicalInformation(Type.MANUAL);
        result.setValue(Field.AUTHOR, "Husni Munaya");
        
        return result;
    }
    
    protected int toSource(int id, StringBuffer buffer) throws Exception {
        int result;
        int i;
        int newID;
        StringBuffer[] subBuffers;
        
        buffer.append("\n");
        buffer.append("  protected static double node" + id + "(Object[] i) {\n");
        
        // leaf?
    if (m_Attribute == null) {
        result = id;
        if (Double.isNaN(m_ClassValue)) {
        buffer.append("    return Double.NaN;");
        } else {
            buffer.append("    return " + m_ClassValue + ";");
        }
        if (m_ClassAttribute != null) {
            buffer.append(" // " + m_ClassAttribute.value((int) m_ClassValue));
        }
        buffer.append("\n");
        buffer.append("  }\n");
    } else {
        buffer.append("    checkMissing(i, " + m_Attribute.index() + ");\n\n");
        buffer.append("    // " + m_Attribute.name() + "\n");
      
        // subtree calls
        subBuffers = new StringBuffer[m_Attribute.numValues()];
        newID = id;
        for (i = 0; i < m_Attribute.numValues(); i++) {
        newID++;

            buffer.append("    ");
            if (i > 0) {
                buffer.append("else ");
            }
            buffer.append("if (((String) i[" + m_Attribute.index() 
                + "]).equals(\"" + m_Attribute.value(i) + "\"))\n");
            buffer.append("      return node" + newID + "(i);\n");

            subBuffers[i] = new StringBuffer();
            newID = m_Successors[i].toSource(newID, subBuffers[i]);
        }
        buffer.append("    else\n");
        buffer.append("      throw new IllegalArgumentException(\"Value '\" + i["
            + m_Attribute.index() + "] + \"' is not allowed!\");\n");
        buffer.append("  }\n");

        // output subtree code
        for (i = 0; i < m_Attribute.numValues(); i++) {
            buffer.append(subBuffers[i].toString());
        }
        subBuffers = null;
      
        result = newID;
    }
    
    return result;
    }
    @Override
    public String toSource(String className) throws Exception {
        StringBuffer result;
        int id;
        
        result = new StringBuffer();
        result.append("class " + className + " {\n");
        result.append("  private static void checkMissing(Object[] i, int index) {\n");
        result.append("    if (i[index] == null)\n");
        result.append("      throw new IllegalArgumentException(\"Null values " + "are not allowed!\");\n");
        result.append("  }\n\n");
        result.append("  public static double classify(Object[] i) {\n");
        id = 0;
        result.append("    return node" + id + "(i);\n");
        result.append("  }\n");
        toSource(id, result);
        result.append("}\n");
        
        return result.toString();
    }
    
}
