package geosglm.ark.cs.cmu.edu;

import java.util.Collections;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

public class Util {

	public static ArrayList<Map.Entry<String, Long>> sortHashMapByValue(HashMap<String, Long> hm) {
		
		// ArrayList<Object> sortedCollocates=new ArrayList<Object>();
		
		// Set entries2=hm.entrySet();
		
		// Map.Entry[] entries=new Map.Entry[entries2.size()];
		
		// Iterator<Map.Entry> it=entries2.iterator();
		// int n=0;
		// while(it.hasNext()) {
		// 	entries[n]=it.next();
		// 	n++;
		// }
	
		// Arrays.sort(entries, new Comparator() {
		// 	public int compare(Object lhs, Object rhs) {
		// 	Map.Entry le = (Map.Entry)lhs;
		// 	Map.Entry re = (Map.Entry)rhs;
		// 	return ((Comparable)re.getValue()).compareTo((Comparable)le.getValue());}}
		// );
				
		// for (int i=0; i<entries.length; i++) {
		// 	Map.Entry<Object, Integer> entry=entries[i];
		// 	sortedCollocates.add(entry.getKey());
		// }

		ArrayList<Map.Entry<String, Long>> sortedCollocates = new ArrayList<Map.Entry<String, Long>>(hm.entrySet());
		Collections.sort(sortedCollocates, new ValueThenKeyComparator<String, Long>());
		
		return sortedCollocates;

	}
}

class ValueThenKeyComparator<K extends Comparable<? super K>, V extends Comparable<? super V>> implements Comparator<Map.Entry<K, V>> {
	public int compare(Map.Entry<K, V> a, Map.Entry<K, V> b) {
		int cmp1 = b.getValue().compareTo(a.getValue());
		if (cmp1 != 0) {
			return cmp1;
		} else {
			return b.getKey().compareTo(a.getKey());
		}
	}
}
