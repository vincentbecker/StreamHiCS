package streams;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import weka.core.Instance;

public class GaussianStreamTest {

	private static final int numInstances = 100;
	private GaussianStream stream;
	
	@Before
	public void setUp() throws Exception {
		double[][] covarianceMatrix = {{0.5, 0}, {0, 0.5}};
		double[] mean = {2,-2};
		stream = new GaussianStream(mean, covarianceMatrix, 1);
	}

	@Test
	public void test() {
		
		for (int i = 0; i < numInstances; i++) {
			Instance inst = stream.nextInstance();
			printInstance(inst);
		}
		
		fail("Not yet implemented");
	}

	private void printInstance(Instance instance){
		String s = "";
		for(int i = 0; i < instance.numAttributes() - 2; i++){
			s += instance.value(i) + ",";
		}
		s += instance.value(instance.numAttributes() - 2);
		System.out.println(s);
	}
}
