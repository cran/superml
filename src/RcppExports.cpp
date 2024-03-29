// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// superSplit
CharacterVector superSplit(std::string str, char sep);
RcppExport SEXP _superml_superSplit(SEXP strSEXP, SEXP sepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type str(strSEXP);
    Rcpp::traits::input_parameter< char >::type sep(sepSEXP);
    rcpp_result_gen = Rcpp::wrap(superSplit(str, sep));
    return rcpp_result_gen;
END_RCPP
}
// superNgrams
std::vector<std::string> superNgrams(std::string str, NumericVector ngram_range, char sep);
RcppExport SEXP _superml_superNgrams(SEXP strSEXP, SEXP ngram_rangeSEXP, SEXP sepSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type str(strSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ngram_range(ngram_rangeSEXP);
    Rcpp::traits::input_parameter< char >::type sep(sepSEXP);
    rcpp_result_gen = Rcpp::wrap(superNgrams(str, ngram_range, sep));
    return rcpp_result_gen;
END_RCPP
}
// superTokenizer
std::vector<std::string> superTokenizer(std::vector<std::string> string);
RcppExport SEXP _superml_superTokenizer(SEXP stringSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<std::string> >::type string(stringSEXP);
    rcpp_result_gen = Rcpp::wrap(superTokenizer(string));
    return rcpp_result_gen;
END_RCPP
}
// superCountMatrix
NumericMatrix superCountMatrix(std::vector<std::string> sent, std::vector<std::string> tokens);
RcppExport SEXP _superml_superCountMatrix(SEXP sentSEXP, SEXP tokensSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<std::string> >::type sent(sentSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type tokens(tokensSEXP);
    rcpp_result_gen = Rcpp::wrap(superCountMatrix(sent, tokens));
    return rcpp_result_gen;
END_RCPP
}
// dot
float dot(NumericVector a, NumericVector b, bool norm);
RcppExport SEXP _superml_dot(SEXP aSEXP, SEXP bSEXP, SEXP normSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type a(aSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type b(bSEXP);
    Rcpp::traits::input_parameter< bool >::type norm(normSEXP);
    rcpp_result_gen = Rcpp::wrap(dot(a, b, norm));
    return rcpp_result_gen;
END_RCPP
}
// dotmat
NumericVector dotmat(NumericVector a, NumericMatrix b, const bool norm);
RcppExport SEXP _superml_dotmat(SEXP aSEXP, SEXP bSEXP, SEXP normSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type a(aSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type b(bSEXP);
    Rcpp::traits::input_parameter< const bool >::type norm(normSEXP);
    rcpp_result_gen = Rcpp::wrap(dotmat(a, b, norm));
    return rcpp_result_gen;
END_RCPP
}
// sorted
std::vector<double> sorted(NumericVector v);
RcppExport SEXP _superml_sorted(SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(sorted(v));
    return rcpp_result_gen;
END_RCPP
}
// sort_index
std::vector<int> sort_index(NumericVector vec, const bool ascending);
RcppExport SEXP _superml_sort_index(SEXP vecSEXP, SEXP ascendingSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< const bool >::type ascending(ascendingSEXP);
    rcpp_result_gen = Rcpp::wrap(sort_index(vec, ascending));
    return rcpp_result_gen;
END_RCPP
}
// normalise2d
arma::mat normalise2d(NumericMatrix mat, const int pnorm, const int axis);
RcppExport SEXP _superml_normalise2d(SEXP matSEXP, SEXP pnormSEXP, SEXP axisSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type mat(matSEXP);
    Rcpp::traits::input_parameter< const int >::type pnorm(pnormSEXP);
    Rcpp::traits::input_parameter< const int >::type axis(axisSEXP);
    rcpp_result_gen = Rcpp::wrap(normalise2d(mat, pnorm, axis));
    return rcpp_result_gen;
END_RCPP
}
// normalise1d
std::vector<double> normalise1d(NumericVector vec, const int pnorm);
RcppExport SEXP _superml_normalise1d(SEXP vecSEXP, SEXP pnormSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type vec(vecSEXP);
    Rcpp::traits::input_parameter< const int >::type pnorm(pnormSEXP);
    rcpp_result_gen = Rcpp::wrap(normalise1d(vec, pnorm));
    return rcpp_result_gen;
END_RCPP
}
// avg_doc_len
double avg_doc_len(std::vector<std::string> ss);
RcppExport SEXP _superml_avg_doc_len(SEXP ssSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<std::string> >::type ss(ssSEXP);
    rcpp_result_gen = Rcpp::wrap(avg_doc_len(ss));
    return rcpp_result_gen;
END_RCPP
}
// idf
double idf(std::string q, std::vector<std::string> corpus);
RcppExport SEXP _superml_idf(SEXP qSEXP, SEXP corpusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type q(qSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type corpus(corpusSEXP);
    rcpp_result_gen = Rcpp::wrap(idf(q, corpus));
    return rcpp_result_gen;
END_RCPP
}
// sort_vector_with_names
NumericVector sort_vector_with_names(NumericVector x);
RcppExport SEXP _superml_sort_vector_with_names(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(sort_vector_with_names(x));
    return rcpp_result_gen;
END_RCPP
}
// bm_25
NumericVector bm_25(std::string document, std::vector<std::string> corpus, const int top_n);
RcppExport SEXP _superml_bm_25(SEXP documentSEXP, SEXP corpusSEXP, SEXP top_nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type document(documentSEXP);
    Rcpp::traits::input_parameter< std::vector<std::string> >::type corpus(corpusSEXP);
    Rcpp::traits::input_parameter< const int >::type top_n(top_nSEXP);
    rcpp_result_gen = Rcpp::wrap(bm_25(document, corpus, top_n));
    return rcpp_result_gen;
END_RCPP
}
// SortOccurence
std::vector<std::string> SortOccurence(std::vector<std::string>& vectors);
RcppExport SEXP _superml_SortOccurence(SEXP vectorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::vector<std::string>& >::type vectors(vectorsSEXP);
    rcpp_result_gen = Rcpp::wrap(SortOccurence(vectors));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_superml_superSplit", (DL_FUNC) &_superml_superSplit, 2},
    {"_superml_superNgrams", (DL_FUNC) &_superml_superNgrams, 3},
    {"_superml_superTokenizer", (DL_FUNC) &_superml_superTokenizer, 1},
    {"_superml_superCountMatrix", (DL_FUNC) &_superml_superCountMatrix, 2},
    {"_superml_dot", (DL_FUNC) &_superml_dot, 3},
    {"_superml_dotmat", (DL_FUNC) &_superml_dotmat, 3},
    {"_superml_sorted", (DL_FUNC) &_superml_sorted, 1},
    {"_superml_sort_index", (DL_FUNC) &_superml_sort_index, 2},
    {"_superml_normalise2d", (DL_FUNC) &_superml_normalise2d, 3},
    {"_superml_normalise1d", (DL_FUNC) &_superml_normalise1d, 2},
    {"_superml_avg_doc_len", (DL_FUNC) &_superml_avg_doc_len, 1},
    {"_superml_idf", (DL_FUNC) &_superml_idf, 2},
    {"_superml_sort_vector_with_names", (DL_FUNC) &_superml_sort_vector_with_names, 1},
    {"_superml_bm_25", (DL_FUNC) &_superml_bm_25, 3},
    {"_superml_SortOccurence", (DL_FUNC) &_superml_SortOccurence, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_superml(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
