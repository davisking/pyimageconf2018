
#include <dlib/python.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/pipe.h>
#include <dlib/simd.h>
#include <thread>
#include <pybind11/stl.h>
#include "cuda_stuff.h"
#include <dlib/cuda/cuda_dlib.h>

namespace py = pybind11;

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

numpy_image<float> coherent_hough_transform (
    const hough_transform& ht,
    const numpy_image<unsigned char>& edges_,
    const numpy_image<float>& horz_,
    const numpy_image<float>& vert_
)
{
    auto edges = make_image_view(edges_);
    auto horz = make_image_view(horz_);
    auto vert = make_image_view(vert_);

    DLIB_CASSERT(have_same_dimensions(edges, horz, vert));

    array2d<matrix<double,2,2>> hcoherent(ht.size(), ht.size());
    for (auto& v : hcoherent)
        v = 0;

    auto record_hit = [&](const point& hough_point, const point& p, float val)
    {
        matrix<double,2,1> v = {horz[p.y()][p.x()], 
                                vert[p.y()][p.x()]};

        hcoherent[hough_point.y()][hough_point.x()] += v*trans(v);
    };
    ht.perform_generic_hough_transform(edges, record_hit);

    numpy_image<float> himg_(ht.nr(), ht.nc());
    auto himg = make_image_view(himg_);
    for (long r = 0; r < himg.nr(); ++r)
    {
        for (long c = 0; c < himg.nc(); ++c)
        {
            matrix<double,0,1> ev = real_eigenvalues(hcoherent[r][c]);
            if (min(ev)/max(ev) < 0.30)
                himg[r][c] = max(ev);
            else
                himg[r][c] = 0;
        }
    }

    return himg_;
}

// ----------------------------------------------------------------------------------------

void discard_wacky_edge_groups (
    numpy_image<float>& edges_,
    const numpy_image<float>& horz_,
    const numpy_image<float>& vert_
)
{
    auto edges = make_image_view(edges_);
    auto horz = make_image_view(horz_);
    auto vert = make_image_view(vert_);

    DLIB_CASSERT(have_same_dimensions(edges, horz, vert));

    double dotprod_angle_thresh = cos(25*pi/180);

    auto connected_if = [&](const auto& img, const point& a, const point& b) {
        dpoint v1(horz[a.y()][a.x()], vert[a.y()][a.x()]);
        dpoint v2(horz[b.y()][b.x()], vert[b.y()][b.x()]);
        return (img[a.y()][a.x()] != 0 && img[b.y()][b.x()] != 0) && dot(v1,v2) > dotprod_angle_thresh;
    };

    array2d<uint32> labels;
    auto num_blobs = label_connected_blobs(edges, zero_pixels_are_background(), neighbors_8(), connected_if, labels);

    matrix<unsigned long,0,1> blob_sizes;
    get_histogram(labels, blob_sizes, num_blobs);

    // blank out short edges
    for (long r = 0; r < labels.nr(); ++r)
        for (long c = 0; c < labels.nc(); ++c)
            if (blob_sizes(labels[r][c]) < 20)
                edges[r][c] = 0;
}

// ----------------------------------------------------------------------------------------

std::vector<rectangle> blobs_to_rects (
    const numpy_image<uint32_t>& labels_,
    size_t num_blobs
)
{
    auto labels = make_image_view(labels_);
    std::vector<rectangle> rects(num_blobs);
    for (long r = 0; r < labels.nr(); ++r)
        for (long c = 0; c < labels.nc(); ++c)
            if (labels[r][c] != 0)
                rects[labels[r][c]] += point(c,r);
    return rects;
}

// ----------------------------------------------------------------------------------------

template <typename T>
void zero_pixels_not_labeled_with_val (
    numpy_image<T>& img_,
    const numpy_image<uint32_t>& labels_,
    uint32_t val
)
{
    auto img = make_image_view(img_);
    auto labels = make_image_view(labels_);
    DLIB_CASSERT(have_same_dimensions(img, labels));

    for (long r = 0; r < img.nr(); ++r)
        for (long c = 0; c < img.nc(); ++c)
            if (labels[r][c] != val)
                img[r][c] = 0;
}

// ----------------------------------------------------------------------------------------

py::list hash_images (
    const py::list& images
)
{
    const matrix<float> random_projections = matrix_cast<float>(gaussian_randm(100,75*75));
    py::list result;
    for (auto& filename : images)
    {
        matrix<float> img,img2,h;

        load_image(img, filename.cast<std::string>());
        img2.set_size(75,75);
        resize_image(img, img2); 

        img2 -= 110;

        h = random_projections*reshape_to_column_vector(img2);
        h = h>0;

        auto thehash = murmur_hash3_128bit(&h(0),h.size()*sizeof(float)).first;
        result.append(py::make_tuple(thehash, filename));
    }

    return result;
}

py::list hash_images_parallel (
    const std::vector<std::string>& images 
)
{
    const matrix<float> random_projections = matrix_cast<float>(gaussian_randm(100,75*75));
    std::mutex m;
    py::list result;
    parallel_for(0, images.size(), [&](long i)
    {
        matrix<float> img,img2,h;

        load_image(img, images[i]);
        img2.set_size(75,75);
        resize_image(img, img2); 

        img2 -= 110;

        h = random_projections*reshape_to_column_vector(img2);
        h = h>0;

        auto thehash = murmur_hash3_128bit(&h(0),h.size()*sizeof(float)).first;

        std::lock_guard<std::mutex> lock(m);
        result.append(py::make_tuple(thehash, images[i]));
    });

    return result;
}

// ----------------------------------------------------------------------------------------

class threaded_data_loader
{
public:
    threaded_data_loader(
        const std::vector<string>& filenames_,
        size_t num_threads,
        size_t buffer_size
    ) : filenames(filenames_), images(buffer_size)
    {
        for (size_t i = 0; i < num_threads; ++i)
        {
            threads.emplace_back([&,i](){
                dlib::rand rnd(i);
                matrix<rgb_pixel> img;
                while(images.is_enabled())
                {
                    try
                    {
                        load_image(img, filenames[rnd.get_integer(filenames.size())]);
                        images.enqueue(img);
                    } catch(exception& e)
                    {
                        cout << e.what() << endl;
                    }
                }
            });
        }
    }

    ~threaded_data_loader()
    {
        // Tell the threads to terminate
        images.disable();
        // and you have to join with thread objects before letting them destruct.  This
        // just waits for them to terminate.
        for (auto& t : threads)
            t.join();
    }

    size_t number_of_threads() const { return threads.size(); }

    numpy_image<rgb_pixel> get_next_image()
    {
        matrix<rgb_pixel> img;
        images.dequeue(img);
        return std::move(img);
    }

private:

    std::vector<string> filenames;
    dlib::pipe<matrix<rgb_pixel>> images;
    std::vector<thread> threads;
};

// ----------------------------------------------------------------------------------------

template <typename T>
double sum_row_major_order (const numpy_image<T>& img_)
{
    auto img = make_image_view(img_);

    double val = 0;
    //for (int iter = 0; iter < 2000; ++iter)
    for (long r = 0; r < img.nr(); ++r)
    {
        for (long c = 0; c < img.nc(); ++c)
        {
            val += img[r][c];
        }
    }
    return val;
}

double sum_row_major_order_simd (const numpy_image<float>& img_)
{
    auto img = make_image_view(img_);

    DLIB_CASSERT(img.size()%8 == 0);

    double result = 0;
    const float* data = &img[0][0];
    //for (int iter = 0; iter < 2000; ++iter)
    {
        simd8f val = 0;
        simd8f temp;
        for (long i = 0; i+7 < img.size(); i+=8)
        {
            temp.load(data+i);
            val += temp;
        }
        result += sum(val);
    }

    return result;
}

template <typename T>
double sum_column_major_order (const numpy_image<T>& img_)
{
    auto img = make_image_view(img_);

    double val = 0;
    //for (int iter = 0; iter < 2000; ++iter)
    for (long c = 0; c < img.nc(); ++c)
    {
        for (long r = 0; r < img.nr(); ++r)
        {
            val += img[r][c];
        }
    }
    return val;
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

void py_cuda_add_value_to_each_element(
    numpy_image<float>& img_,
    const float value
)
{
    auto img = make_image_view(img_);
    cuda_data_ptr<float> temp(img.size());
    // copy to the GPU
    memcpy(temp, &img[0][0]);

    cuda_add_value_to_each_element(temp, value);

    // copy from GPU back to CPU
    memcpy(&img[0][0], temp);
}

// ----------------------------------------------------------------------------------------

cuda_data_ptr<float> numpy_to_cuda (
    const numpy_image<float>& arr_
)
{
    auto arr = make_image_view(arr_);

    cuda_data_ptr<float> temp(arr.size());
    memcpy(temp, &arr[0][0]);
    return temp;
}

numpy_image<float> cuda_to_numpy (
    const cuda_data_ptr<float>& arr
)
{
    numpy_image<float> temp_(arr.size(),1);
    auto temp = make_image_view(temp_);

    memcpy(&temp[0][0], arr);
    return temp_;
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

void raster_scan(
    const numpy_image<float>& img_,
    numpy_image<float>& dist_,
    numpy_image<float>& lower_,
    numpy_image<float>& upper_
)
{
    auto img = make_image_view(img_);
    auto dist = make_image_view(dist_);
    auto lower = make_image_view(lower_);
    auto upper = make_image_view(upper_);
    DLIB_CASSERT(have_same_dimensions(img,dist,lower,upper));

    auto area = shrink_rect(get_rect(img),1);

    auto check_neighbor = [&](long r, long c, long neighbor_r, long neighbor_c) 
    {
        auto l = std::min(lower[neighbor_r][neighbor_c], img[r][c]);
        auto u = std::max(upper[neighbor_r][neighbor_c], img[r][c]);
        auto d = u-l;
        if (d < dist[r][c])
        {
            lower[r][c] = l;
            upper[r][c] = u;
            dist[r][c] = d;
        }
    };

    // scan top to bottom
    for (long r = area.top(); r <= area.bottom(); ++r)
    {
        for (long c = area.left(); c <= area.right(); ++c)
        {
            check_neighbor(r,c, r-1,c);
            check_neighbor(r,c, r,c-1);
        }
    }

    // scan bottom to top
    for (long r = area.bottom(); r >= area.top(); --r)
    {
        for (long c = area.right(); c >= area.left(); --c)
        {
            check_neighbor(r,c, r+1,c);
            check_neighbor(r,c, r,c+1);
        }
    }


    // scan left to right 
    for (long c = area.left(); c <= area.right(); ++c)
    {
        for (long r = area.top(); r <= area.bottom(); ++r)
        {
            check_neighbor(r,c, r-1,c);
            check_neighbor(r,c, r,c-1);
        }
    }

    // scan right to left 
    for (long c = area.right(); c >= area.left(); --c)
    {
        for (long r = area.bottom(); r >= area.top(); --r)
        {
            check_neighbor(r,c, r+1,c);
            check_neighbor(r,c, r,c+1);
        }
    }
}

PYBIND11_MODULE(superfast, m)
{

    m.attr("__version__") = "1.2.3";
    m.attr("__time_compiled__") = std::string(__DATE__) + " " + std::string(__TIME__);

    m.def("hash_images", hash_images, py::arg("image_filenames"));
    m.def("hash_images_parallel", hash_images_parallel, py::arg("image_filenames"));

    m.def("blobs_to_rects", blobs_to_rects, 
        py::arg("labels"), py::arg("num_blobs"),
        "a doc string that you should definitely write");




    m.def("coherent_hough_transform", coherent_hough_transform, 
        py::arg("ht"), py::arg("edges"), py::arg("horz"), py::arg("vert"),
        "a doc string that you should definitely write");


    m.def("discard_wacky_edge_groups", discard_wacky_edge_groups, 
        py::arg("edges"), py::arg("horz"), py::arg("vert"),
        "a doc string that you should definitely write");

    m.def("zero_pixels_not_labeled_with_val", zero_pixels_not_labeled_with_val<float>, 
        py::arg("img"), py::arg("labels"), py::arg("val"),
        "a doc string that you should definitely write");
    m.def("zero_pixels_not_labeled_with_val", zero_pixels_not_labeled_with_val<uint8_t>, 
        py::arg("img"), py::arg("labels"), py::arg("val"),
        "a doc string that you should definitely write");



    

    
    m.def("raster_scan", raster_scan, py::arg("img"), py::arg("dist"), py::arg("lower"), py::arg("upper"));






    m.def("sum_row_major_order", sum_row_major_order<float>);
    m.def("sum_row_major_order", sum_row_major_order<double>);
    m.def("sum_row_major_order_simd", sum_row_major_order_simd);
    m.def("sum_column_major_order", sum_column_major_order<float>);
    m.def("sum_column_major_order", sum_column_major_order<double>);





    py::class_<threaded_data_loader>(m, "threaded_data_loader", "doc string about this object")
        .def(py::init<std::vector<string>,size_t,size_t>(), py::arg("filenames"), py::arg("num_threads"), py::arg("buffer_size")=3)
        .def("get_next_image", &threaded_data_loader::get_next_image)
        .def_property_readonly("number_of_threads", &threaded_data_loader::number_of_threads);






    m.def("cuda_add_value_to_each_element", py_cuda_add_value_to_each_element, py::arg("img"), py::arg("value"));

    py::class_<cuda_data_ptr<float>>(m, "cuda_data", "An array of float32 data on the GPU")
        .def(py::init<size_t>(), "Allocate an array off N float32s on the GPU", py::arg("N"))
        .def(py::init<>(&numpy_to_cuda))
        .def("to_numpy", cuda_to_numpy, "convert this cuda array to a numpy array.")
        .def_property_readonly("size", &cuda_data_ptr<float>::size);

    m.def("cuda_add_value_to_each_element", cuda_add_value_to_each_element, py::arg("img"), py::arg("value"));
    m.def("cuda_add_value_to_each_element_simple", cuda_add_value_to_each_element_simple, py::arg("img"), py::arg("value"));
    m.def("cuda_add_value_to_each_element_ugly", cuda_add_value_to_each_element_ugly, py::arg("img"), py::arg("value"));

    m.def("cuda_dot_product", cuda_dot_product, py::arg("out"), py::arg("a"), py::arg("b"));
    m.def("cuda_matrix_vector_multiply", cuda_matrix_vector_multiply, py::arg("out"), py::arg("M"), py::arg("v"), "compute: out = M*v");

    m.def("cuda_set_device", [](int id){set_device(id); }, py::arg("device_id"));
    m.def("cuda_device_synchronize", [](int id){device_synchronize(id); }, py::arg("device_id"));

}

