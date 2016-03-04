package streamhics_realworlddata;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealMatrix;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.streams.ArffFileStream;
import streamdatastructures.MCAdapter;
import streamdatastructures.CorrelationSummary;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class IntrusionDetection {

	private static Stopwatch stopwatch;
	
	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}
	
	@AfterClass
	public static void calculateAverageScores() {
		System.out.println(stopwatch.toString());
	}

	/*
	@Test
	public void findAttackPeriods() {
		String path = "Tests/RealWorldData/kddcup99_10_percent_filtered.arff";
		String outPath = "Tests/RealWorldData/kddcup99_attackPeriods.txt";
		// Class index is last attribute but not relevant for this task
		ArffFileStream stream = new ArffFileStream(path, -1);

		List<String> attacks = new ArrayList<String>();
		int numberSample = 0;
		boolean attackStarted = false;
		int attackBeginning = 0;
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			String classLabel = inst.stringValue(inst.classIndex());
			if(!classLabel.equals("normal")){
				if(!attackStarted){
					//attacks.add("Beginning: " + numberSample);
					attackStarted = true;
					attackBeginning = numberSample;
				}
			}else{
				if(attackStarted){
					if(numberSample - attackBeginning >= 100){
						attacks.add("Beginning: " + attackBeginning);
						attacks.add("End: " + numberSample);
					}
					attackStarted = false;
				}
			}
			numberSample++;
		}
		try {
			Files.write(Paths.get(outPath), attacks);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	*/
	
	@Test
	public void testCorrBeforeAttacks() {
		String path = "Tests/RealWorldData/kddcup99_10_percent_filtered.arff";
		String attacksPath = "Tests/RealWorldData/kddcup99_attackPeriods.txt";
		// Class index is last attribute but not relevant for this task
		ArffFileStream stream = new ArffFileStream(path, -1);
		
		List<String> attacks = null;
		try {
			attacks = Files.readAllLines(Paths.get(attacksPath));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double[] attackPoints = new double[attacks.size()];
		int i = 0;
		for(String s : attacks){
			String[] split = s.split(" ");
			attackPoints[i] = Double.parseDouble(split[1]);
			i++;
		}
		int numberOfDimensions = 23;
		int m = 50;
		double alpha = 0.1;
		double epsilon = 0.1;
		double threshold = 0.6;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 1000;
		int numberSamples = 0;
		MCAdapter adapter = new MCAdapter(horizon, 20, 0.1, "radius");
		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);
		
		CorrelationSummary correlationSummary = null;
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff,
				contrastEvaluator, correlationSummary);
		ChangeChecker changeChecker = new TimeCountChecker(1000000);
		Callback callback = new Callback() {
			@Override
			public void onAlarm() {
				//System.out.println("StreamHiCS: onAlarm() at " + numberSamples);
			}
		};
		StreamHiCS streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);
		
		int attackCounter = 0;
		ArrayList<String> numberElements = new ArrayList<String>();
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			/*
			if(numberSamples == attackPoints[attackCounter]){
				streamHiCS.evaluateCorrelatedSubspaces();
				System.out.println("Number samples: " + numberSamples);
				System.out.println("Number elements: " + streamHiCS.getNumberOfElements());
				System.out.println(streamHiCS.toString());
				attackCounter++;
			}
			*/
			streamHiCS.add(inst);
			if(numberSamples % 10 == 0){
				numberElements.add(numberSamples + "," + streamHiCS.getNumberOfElements());
			}
			numberSamples++;
		}
		String filePath = "C:/Users/Vincent/Desktop/IntrusionDetection_numberElements.csv";

		try {
			Files.write(Paths.get(filePath), numberElements);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
