import java.util.ArrayList;

import StreamDataStructures.SelfOrganizingMapContainer;
import weka.core.Attribute;
import weka.core.Instance;
import moa.streams.generators.RandomRBFGenerator;

public class Main {

	public static void main(String[] args) {
		int numInstances = 1000;
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();
		int numberOfDimensions = stream.numAttsOption.getValue();

		StreamHiCS streamHiCS = new StreamHiCS(numberOfDimensions, 100);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			System.out.println(inst.value(0));
			streamHiCS.add(inst);
			numberSamples++;
			// System.out.println(streamHiCS.getNumberOfInstances());
		}
		
		/*
		int numInstances = 10000;
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();
		ArrayList<Attribute> a = new ArrayList<Attribute>();
		for(int i = 0; i< stream.getHeader().numAttributes(); i++){
			a.add(stream.getHeader().attribute(i));
		}
		SelfOrganizingMapContainer som = new SelfOrganizingMapContainer(a);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			System.out.println(inst.value(0));
			som.add(inst);
			numberSamples++;
			// System.out.println(streamHiCS.getNumberOfInstances());
		}
		
		som.trainSOM();
		System.out.println(som.getNumberOfInstances());
		*/
	}
}
