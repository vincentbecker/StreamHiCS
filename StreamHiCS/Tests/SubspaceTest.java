import static org.junit.Assert.*;
import org.junit.Test;

import streamDataStructures.Subspace;

public class SubspaceTest {

	@Test
	public void mergeTest1() {
		Subspace s1 = new Subspace(0);
		Subspace s2 = new Subspace(0);
		Subspace mergeResult = Subspace.merge(s1, s2);
		Subspace correctResult = new Subspace(0);
		assertTrue(mergeResult.equals(correctResult));
		;
	}

	@Test
	public void mergeTest2() {
		Subspace s1 = new Subspace(0, 1);
		Subspace s2 = new Subspace(0, 2);
		Subspace mergeResult = Subspace.merge(s1, s2);
		Subspace correctResult = new Subspace(0, 1, 2);
		assertTrue(mergeResult.equals(correctResult));
		;
	}

	@Test
	public void mergeTest3() {
		Subspace s1 = new Subspace(1, 2, 3);
		Subspace s2 = new Subspace(1, 2, 4);
		Subspace mergeResult = Subspace.merge(s1, s2);
		Subspace correctResult = new Subspace(1, 2, 3, 4);
		assertTrue(mergeResult.equals(correctResult));
		;
	}

	@Test
	public void mergeTest4() {
		Subspace s1 = new Subspace(1, 2, 4);
		Subspace s2 = new Subspace(1, 2, 3);
		Subspace mergeResult = Subspace.merge(s1, s2);
		Subspace correctResult = new Subspace(1, 2, 3, 4);
		assertTrue(mergeResult.equals(correctResult));
		;
	}

	@Test
	public void mergeTest5() {
		Subspace s1 = new Subspace(0);
		Subspace s2 = new Subspace(0, 1);
		Subspace mergeResult = Subspace.merge(s1, s2);
		assertTrue(mergeResult == null);
		;
	}

	@Test
	public void mergeTest6() {
		Subspace s1 = new Subspace(0, 1);
		Subspace s2 = new Subspace(1, 2);
		Subspace mergeResult = Subspace.merge(s1, s2);
		assertTrue(mergeResult == null);
		;
	}

	@Test
	public void mergeTest7() {
		Subspace s1 = new Subspace(3, 4, 5, 6, 7);
		Subspace s2 = new Subspace(3, 4, 5, 6, 8);
		Subspace mergeResult = Subspace.merge(s1, s2);
		Subspace correctResult = new Subspace(3, 4, 5, 6, 7, 8);
		assertTrue(mergeResult.equals(correctResult));
		;
	}

	@Test
	public void mergeTest8() {
		Subspace s1 = new Subspace();
		Subspace s2 = new Subspace();
		Subspace mergeResult = Subspace.merge(s1, s2);
		assertTrue(mergeResult == null);
		;
	}

	@Test
	public void subspaceOfTest1() {
		Subspace s1 = new Subspace();
		assertTrue(s1.isSubspaceOf(s1));
	}

	@Test
	public void subspaceOfTest2() {
		Subspace s1 = new Subspace(0);
		Subspace s2 = new Subspace(0, 1);
		assertTrue(s1.isSubspaceOf(s2));
	}

	@Test
	public void subspaceOfTest3() {
		Subspace s1 = new Subspace(0);
		Subspace s2 = new Subspace(1);
		assertTrue(!s1.isSubspaceOf(s2));
	}

	@Test
	public void subspaceOfTest4() {
		Subspace s1 = new Subspace(0, 1);
		Subspace s2 = new Subspace(1);
		assertTrue(!s1.isSubspaceOf(s2));
	}

	@Test
	public void subspaceOfTest5() {
		Subspace s1 = new Subspace(0, 4);
		Subspace s2 = new Subspace(0, 2, 4);
		assertTrue(s1.isSubspaceOf(s2));
	}
}
