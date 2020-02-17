#ifndef CLASSIFIER_UTIL_H
#define CLASSIFIER_UTIL_H

#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include "matplotlib.h"

/**** type definitions ****/
typedef std::string FilePath;                               // file path
typedef std::vector<FilePath> FileList;                     // list of file paths
typedef std::vector<std::string> WordList;                  // list of words (with repetition and unsorted)
typedef std::unordered_map<std::string, double> ProbDict;   // dictionary of probabilities
typedef std::unordered_map<std::string, size_t> FreqDict;   // dictionary of frequencies
enum CLASS {SPAM, HAM};

namespace plt = matplotlibcpp;

/**** function prototypes ****/
std::vector<std::string> get_files_in_folder(FilePath folder_path, std::string extension = ".txt");
WordList get_words_in_file(FilePath file_path);
FreqDict get_word_freq(FileList files);
void plot_probabilities(ProbDict& prob_dict);

/**** functions ****/

void plot_probabilities(ProbDict& prob_dict)
{
//    std::vector<double> x, y;
//    size_t lbl_cnt = 1;
//
//    // plot all map points
//    for (const Point& p : pnt_arr)
//    {
//        x.push_back(p[0]);
//        y.push_back(p[1]);
//    }
//    plt::scatter(x, y, 3);
//
//    // plot k nearest points and annotate
//    x.clear();
//    y.clear();
//    for (const Point& p : k_arr)
//    {
//        x.push_back(p[0]);
//        y.push_back(p[1]);
//        plt::annotate(std::to_string(lbl_cnt), p[0], p[1]);
//        lbl_cnt++;
//    }
//    plt::scatter(x, y, 15);
//
//    // plot input point and annotate
//    plt::annotate("IN", usr_pnt[0], usr_pnt[1]);
//    plt::scatter(std::vector<double>{usr_pnt[0]}, std::vector<double>{usr_pnt[1]}, 20);
//
//    plt::title("kNN Search Result");
//    plt::show();
}

#endif //CLASSIFIER_UTIL_H
