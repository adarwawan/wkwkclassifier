/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.wkwk.classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


/**
 *
 * @author Husni
 */
public class Id3 
        extends Classifier
        implements TechnicalInformationHandler, Sourcable {

    @Override
    public void buildClassifier(Instances i) {
        
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
    
        result = new TechnicalInformation(Type.MANUAL);
        result.setValue(Field.AUTHOR, "Husni Munaya");
        
        return result;
    }

    @Override
    public String toSource(String string) {
        return "";
    }
    
}
