import weka.core.DenseInstance;
import weka.core.Instance;

import java.awt.EventQueue;
import drawing.PointsEx;
import moa.streams.generators.RandomRBFGenerator;
import streamDataStructures.SelfOrganizingMap;

public class Main {	
	
	public static void main(String[] args) {
			
		EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {

                PointsEx ex = new PointsEx();
                ex.setVisible(true);
            }
        });
		
		/*
		 * int numInstances = 100000; // RandomRBFGenerator stream = new
		 * RandomRBFGenerator(); // stream.prepareForUse(); // int
		 * numberOfDimensions = stream.numAttsOption.getValue(); //
		 * CorrelatedStream stream = new CorrelatedStream(); double[][]
		 * covariances = { { 1, 0.9, 0, 0, 0 }, { 0.9, 1, 0, 0, 0 }, { 0, 0, 1,
		 * 0.5, 0.3 }, { 0, 0, 0.5, 1, 0.3 }, { 0, 0, 0.3, 0.3, 1 } };
		 * GaussianStream stream = new GaussianStream(covariances); int
		 * numberOfDimensions = stream.getNumberOfDimensions();
		 * 
		 * StreamHiCS streamHiCS = new StreamHiCS(numberOfDimensions, 10000, 20,
		 * 0.4, 0.1, 0.8, 10);
		 * 
		 * int numberSamples = 0; while (stream.hasMoreInstances() &&
		 * numberSamples < numInstances) { Instance inst =
		 * stream.nextInstance(); streamHiCS.add(inst); numberSamples++; //
		 * System.out.println(streamHiCS.getNumberOfInstances()); }
		 */

		/*
		int numInstances = 1000;
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();
		SelfOrganizingMap som = new SelfOrganizingMap(stream.getHeader().numAttributes(), 25);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			som.add(inst);
			numberSamples++;
		}

		som.trainSOM();
		System.out.println("Number of clusters: " + som.getNumberOfInstances());
		*/
	}
}
