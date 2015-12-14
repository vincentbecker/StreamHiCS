package streamtests;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.CoresetContrast;
import contrast.MicroclusterContrast;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import clustree.ClusTree;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.clustream.Clustream;
import moa.streams.ArffFileStream;
import streamDataStructures.WithDBSCAN;
import subspace.Subspace;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

/**
 * 
 * 
 * @author Vincent
 *
 */
public class Waveform40_sorted {
	private String path;
	private ArffFileStream stream;
	private int numberSamples = 0;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm() at " + numberSamples);
		}
	};
	private StreamHiCS streamHiCS;
	
	@Test
	public void WaveformSorted() {
		// The change points in the data are: 33368, 66707
		path = "Tests/waveform_sorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 40;
		int m = 50;
		double alpha = 0.25;
		double epsilon = 0.1;
		double threshold = 0.65;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 10000;
		int checkCount = 10000;
		
		System.out.println("Waveform sorted");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	
	private void carryOutTest(int numberOfDimensions, int m, double alpha, double epsilon, double threshold, int cutoff,
			double pruningDifference, int horizon, int checkCount) {
		
		/*
		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(horizon);
		mcs.prepareForUse();
		*/
		
		/*
		WithDBSCAN mcs = new WithDBSCAN();
		mcs.speedOption.setValue(1000);
		mcs.epsilonOption.setValue(1.8);
		mcs.betaOption.setValue(0.2);
		mcs.muOption.setValue(10);
		mcs.lambdaOption.setValue(0.05);
		//mcs.initPointsOption.setValue(100);
		mcs.prepareForUse();
		*/
		/*
		Clustream mcs = new Clustream();
		mcs.kernelRadiFactorOption.setValue(2);
		mcs.maxNumKernelsOption.setValue(500);
		mcs.prepareForUse();
		
		Contrast contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);
		*/
		
		
		//Contrast contrastEvaluator = new SlidingWindowContrast(numberOfDimensions, m, alpha, 10000);
		Contrast contrastEvaluator = new CoresetContrast(m, alpha, 100000, 1000);
		
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		
		// SubspaceBuilder subspaceBuilder = new
		// FastBuilder(covarianceMatrix.length, threshold, pruningDifference,
		// contrastEvaluator);

		ChangeChecker changeChecker = new TimeCountChecker(checkCount);
		//ChangeChecker changeChecker = new FullSpaceContrastChecker(checkCount, numberOfDimensions, contrastEvaluator, 0.2, 0.1);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, callback, null);
		changeChecker.setCallback(streamHiCS);
		
		PearsonsCorrelation pc = new PearsonsCorrelation();
		double[][] data = new double[checkCount][numberOfDimensions];
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ElectricityNorthWest/ENWCorrelations.csv";
		
		List<String> correlOut = new ArrayList<String>();
		double overallMinCorrelation = Double.MAX_VALUE;
		double sumCorrelation = 0;
		int correlCount = 0;
		
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			//mcs.trainOnInstance(inst);
			data[numberSamples % checkCount] = inst.toDoubleArray();
			numberSamples++;
			if (numberSamples % checkCount == 0) {
				//System.out.println("Time: " + numberSamples + ", Number of elements: " + mcs.getMicroClusteringResult().size());
				RealMatrix correlationMatrix = pc.computeCorrelationMatrix(data);
				double[][] cm = correlationMatrix.getData();
				for(int i = 0; i < numberOfDimensions; i++){
					String line = "";
					for(int j = 0; j < numberOfDimensions - 1; j++){
						line += cm[i][j] + ",";
					}
					line += cm[i][numberOfDimensions - 1];
					correlOut.add(line);
				}
				correlOut.add("");
				try {
					Files.write(Paths.get(filePath), correlOut, StandardOpenOption.APPEND);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				correlOut.clear();
				
				data = new double[checkCount][numberOfDimensions];
				System.out.println("Number of elements: " + streamHiCS.getNumberOfElements());
				System.out.println("Correlated: " + streamHiCS.toString());
				if (!streamHiCS.getCurrentlyCorrelatedSubspaces().isEmpty()) {
					for (Subspace s : streamHiCS.getCurrentlyCorrelatedSubspaces().getSubspaces()) {
						double minCorrelation = Double.MAX_VALUE;
						for(int i = 0; i < s.size(); i++){
							for(int j = i + 1; j < s.size(); j++){
								double correl = Math.abs(cm[s.getDimension(i)][s.getDimension(j)]);
								if(!Double.isNaN(correl)){
									sumCorrelation += correl;
								}
								correlCount++;
								if(correl < minCorrelation){
									minCorrelation = correl;
								}
							}
						}
						if(minCorrelation < overallMinCorrelation){
							overallMinCorrelation = minCorrelation;
						}
						System.out.print(s.getContrast() + "|" + minCorrelation + ", ");
					}
					System.out.println();
				}
			}
		}
		double averageCorrelation = sumCorrelation / correlCount;
		System.out.println("Overall minimum abs. correlation: " + overallMinCorrelation + ", average abs. correlation: " + averageCorrelation);
	}

}
