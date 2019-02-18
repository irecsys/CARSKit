package carskit.eval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Measures extends happy.coding.math.Measures {
    public Measures(){
        super();
    }

    public static <T> List<T> getTopNList(List<T> rankedList, int n){
        n= (n>rankedList.size())?rankedList.size():n;
        List<T> newList=new ArrayList<T>(n);
        for(int i=0;i<n;++i){
            newList.add(rankedList.get(i));
        }
        return newList;
    }
    public static <T> double nDCGAt(List<T> rankedList, List<T> groundTruth, int n) {
        return nDCG(getTopNList(rankedList,n), groundTruth);
    }

    public static <T> Map<Integer, Double> nDCGAt(List<T> rankedList, List<T> groundTruth, List<Integer> ns) {
        Map<Integer, Double> ndcg_at_n = new HashMap<>();
        for (int n : ns)
            ndcg_at_n.put(n, nDCGAt(rankedList, groundTruth, n));

        return ndcg_at_n;
    }

    public static <T> double AUCAt(List<T> rankedList, List<T> groundTruth, int numDropped, int n) {
        return AUC(getTopNList(rankedList,n), groundTruth, numDropped);
    }

    public static <T> Map<Integer, Double> AUCAt(List<T> rankedList, List<T> groundTruth, int numDropped, List<Integer> ns) {
        Map<Integer, Double> AUC_at_n = new HashMap<>();
        for (int n : ns)
            AUC_at_n.put(n, AUCAt(rankedList, groundTruth, numDropped,n));

        return AUC_at_n;
    }

    public static <T> double APAt(List<T> rankedList, List<T> groundTruth, int n) {
        return AP(getTopNList(rankedList,n), groundTruth);
    }

    public static <T> Map<Integer, Double> APAt(List<T> rankedList, List<T> groundTruth, List<Integer> ns) {
        Map<Integer, Double> AP_at_n = new HashMap<>();
        for (int n : ns)
            AP_at_n.put(n, APAt(rankedList, groundTruth, n));

        return AP_at_n;
    }

    public static <T> double RRAt(List<T> rankedList, List<T> groundTruth, int n) {
        return RR(getTopNList(rankedList,n), groundTruth);
    }

    public static <T> Map<Integer, Double> RRAt(List<T> rankedList, List<T> groundTruth, List<Integer> ns) {
        Map<Integer, Double> rr_at_n = new HashMap<>();
        for (int n : ns)
            rr_at_n.put(n, RRAt(rankedList, groundTruth, n));

        return rr_at_n;
    }
}
