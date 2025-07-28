library(shiny)
library(ggplot2)
library(dplyr)
library(tools)
library(stringr)
library(scales)

# ---------- prettify filenames ----------
format_label <- function(filename) {
  name  <- file_path_sans_ext(filename)
  parts <- unlist(strsplit(name, "_"))
  parts <- tolower(parts) |> tools::toTitleCase()
  parts[1] <- toupper(parts[1])
  paste(parts, collapse = " ")
}

# ---------- load CSV ----------
load_selected_experiments <- function() {
  csv_dir <- file.path("..", "results", "csv_files")
  keep    <- c("nhanes_experiment_2.csv", "nhanes_experiment_4.csv")
  files   <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)
  files   <- files[basename(files) %in% keep]
  df_list <- lapply(files, function(f) {
    df <- read.csv(f, stringsAsFactors = FALSE)
    df$source_file <- basename(f)
    df
  })
  bind_rows(df_list)
}

# ---------- accuracy (full) ----------
accuracy_score_plus_match <- function(df, method) {
  score_col <- paste0("score_", method)
  match_col <- paste0("match_", method)
  if (!(score_col %in% names(df)) || !(match_col %in% names(df)))
    return(data.frame(threshold = numeric(), accuracy = numeric()))
  thresholds <- if (method == "fuzzy") seq(0, 100, 1) else seq(0, 1, 0.01)
  scores <- df[[score_col]]
  matches <- df[[match_col]]
  gold <- df$target_desc
  labels <- df$label
  acc <- sapply(thresholds, function(t) {
    pred <- ifelse(is.na(scores), 0L, as.integer(scores >= t))
    correct <- ((labels == 0L) & (pred == 0L)) |
      ((labels == 1L) & (pred == 1L) & (matches == gold))
    mean(correct, na.rm = TRUE)
  })
  data.frame(threshold = thresholds, accuracy = acc)
}

# ---------- confusion matrix ----------
compute_confusion_matrix <- function(df, method, threshold) {
  score_col <- paste0("score_", method)
  if (!(score_col %in% names(df)))
    return(matrix(NA, 2, 2, dimnames = list(c("Actual: 1", "Actual: 0"), c("Pred: 1", "Pred: 0"))))
  scores <- df[[score_col]]
  labels <- df$label
  pred   <- ifelse(is.na(scores), 0L, as.integer(scores >= threshold))
  TP <- sum(labels == 1 & pred == 1, na.rm = TRUE)
  FN <- sum(labels == 1 & pred == 0, na.rm = TRUE)
  FP <- sum(labels == 0 & pred == 1, na.rm = TRUE)
  TN <- sum(labels == 0 & pred == 0, na.rm = TRUE)
  matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE,
         dimnames = list(c("Actual: 1", "Actual: 0"), c("Pred: 1", "Pred: 0")))
}

# ---------- metrics summary ----------
conf_subtitle <- function(df, method, threshold) {
  score_col <- paste0("score_", method)
  match_col <- paste0("match_", method)
  if (!(score_col %in% names(df)) || !(match_col %in% names(df)))
    return("Accuracy unavailable")
  scores  <- df[[score_col]]
  matches <- df[[match_col]]
  labels  <- df$label
  gold    <- df$target_desc
  pred    <- ifelse(is.na(scores), 0L, as.integer(scores >= threshold))
  acc_class <- mean(pred == labels, na.rm = TRUE)
  is_correct <- ((labels == 0L) & (pred == 0L)) |
    ((labels == 1L) & (pred == 1L) & (matches == gold))
  acc_full <- mean(is_correct, na.rm = TRUE)
  fnr <- mean((labels == 1L) & (pred == 0L), na.rm = TRUE)
  sprintf("Class Acc: %.1f%% — Full Acc: %.1f%% — FNR: %.1f%%",
          100 * acc_class, 100 * acc_full, 100 * fnr)
}

# ---------- UI ----------
ui <- fluidPage(
  titlePanel("Threshold vs Accuracy — NHANES Experiments"),
  fluidRow(
    column(3, selectInput("experiment", "Select Experiment:", choices = NULL, width = "100%")),
    column(3, selectInput("method", "Select Method:", choices = c("fuzzy", "tfidf", "embed"),
                          selected = "fuzzy", width = "100%")),
    column(3, radioButtons("plot_mode", "Hide:",
                           choices = c("Overlay", "Stacked", "Confusion Matrix Only"),
                           selected = "Overlay", inline = TRUE))
  ),
  fluidRow(
    column(4, uiOutput("fuzzy_slider")),
    column(4, uiOutput("tfidf_slider")),
    column(4, uiOutput("embed_slider"))
  ),
  fluidRow(
    column(12, textOutput("accuracy_summary"), tags$hr())
  ),
  conditionalPanel("input.plot_mode == 'Overlay'",
                   fluidRow(column(12, plotOutput("overlayPlot", height = "450px")))),
  conditionalPanel("input.plot_mode == 'Stacked'",
                   fluidRow(
                     column(12, plotOutput("scorePlot", height = "350px")),
                     column(12, plotOutput("fullPlot",  height = "350px"))
                   )),
  fluidRow(
    column(4, uiOutput("conf_fuzzy")),
    column(4, uiOutput("conf_tfidf")),
    column(4, uiOutput("conf_embed"))
  )
)

# ---------- Server ----------
server <- function(input, output, session) {
  experiment_data <- load_selected_experiments()
  file_choices <- unique(experiment_data$source_file)
  names(file_choices) <- sapply(file_choices, format_label)
  updateSelectInput(session, "experiment", choices = file_choices)
  
  selected_data <- reactive({
    req(input$experiment)
    experiment_data %>% filter(source_file == input$experiment)
  })
  
  get_best_thresh <- function(df, method) {
    acc_df <- accuracy_score_plus_match(df, method)
    acc_df$threshold[which.max(acc_df$accuracy)]
  }
  
  output$fuzzy_slider <- renderUI({
    df <- selected_data()
    default <- get_best_thresh(df, "fuzzy")
    sliderInput("fuzzy_thresh", "Fuzzy Threshold:", min = 0, max = 100, value = default)
  })
  
  output$tfidf_slider <- renderUI({
    df <- selected_data()
    default <- get_best_thresh(df, "tfidf")
    sliderInput("tfidf_thresh", "TFIDF Threshold:", min = 0, max = 1, value = default, step = 0.01)
  })
  
  output$embed_slider <- renderUI({
    df <- selected_data()
    default <- get_best_thresh(df, "embed")
    sliderInput("embed_thresh", "Embed Threshold:", min = 0, max = 1, value = default, step = 0.01)
  })
  
  get_thresh <- function(method) {
    if (method == "fuzzy") input$fuzzy_thresh
    else if (method == "tfidf") input$tfidf_thresh
    else input$embed_thresh
  }
  
  accuracy_score_only <- function(df, method) {
    score_col <- paste0("score_", method)
    if (!(score_col %in% names(df))) return(data.frame(threshold = numeric(), accuracy = numeric()))
    thresholds <- if (method == "fuzzy") seq(0, 100, 1) else seq(0, 1, 0.01)
    scores <- df[[score_col]]
    labels <- df$label
    acc <- sapply(thresholds, function(t) {
      pred <- ifelse(is.na(scores), 0L, as.integer(scores >= t))
      mean(pred == labels, na.rm = TRUE)
    })
    data.frame(threshold = thresholds, accuracy = acc)
  }
  
  acc_score <- reactive({
    accuracy_score_only(selected_data(), input$method)
  })
  
  acc_full <- reactive({
    accuracy_score_plus_match(selected_data(), input$method)
  })
  
  output$accuracy_summary <- renderText({
    df1 <- acc_score()
    df2 <- acc_full()
    if (nrow(df1) == 0 || nrow(df2) == 0) return("No accuracy data available.")
    t_max <- if (input$method == "fuzzy") 100 else 1
    pretty_pair <- function(a0) {
      a0_pct <- round(a0 * 100, 1)
      a1_pct <- 100 - a0_pct
      sprintf("%.1f%% | T%s: %.1f%%", a0_pct, t_max, a1_pct)
    }
    txt_score <- sprintf("Score‑only accuracy — T0: %s", pretty_pair(df1$accuracy[df1$threshold == 0]))
    txt_full  <- sprintf("Score + match accuracy — T0: %s", pretty_pair(df2$accuracy[df2$threshold == 0]))
    paste(txt_score, txt_full, sep = "    ||    ")
  })
  
  output$overlayPlot <- renderPlot({
    df <- bind_rows(
      acc_score() |> mutate(metric = "Score only"),
      acc_full()  |> mutate(metric = "Score + match")
    )
    if (nrow(df) == 0) { plot.new(); title("No data"); return() }
    x_breaks <- if (input$method == "fuzzy") seq(0, 100, 10) else seq(0, 1, 0.1)
    x_limits <- if (input$method == "fuzzy") c(0, 100) else c(0, 1)
    ggplot(df, aes(threshold, accuracy, colour = metric)) +
      geom_line(linewidth = 1.2) +
      scale_color_manual(values = c("steelblue", "firebrick")) +
      scale_x_continuous(limits = x_limits, breaks = x_breaks) +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(
        title = paste("Threshold vs Accuracy —", tools::toTitleCase(input$method)),
        subtitle = input$experiment, x = "Threshold", y = "Accuracy", colour = NULL
      ) +
      theme_classic(base_size = 14) +
      theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5), legend.position = "top")
  })
  
  make_single_plot <- function(df, title_suffix) {
    if (nrow(df) == 0) { plot.new(); title("No data"); return(invisible()) }
    x_breaks <- if (input$method == "fuzzy") seq(0, 100, 10) else seq(0, 1, 0.1)
    x_limits <- if (input$method == "fuzzy") c(0, 100) else c(0, 1)
    ggplot(df, aes(threshold, accuracy)) +
      geom_line(linewidth = 1.2, colour = "steelblue") +
      scale_x_continuous(limits = x_limits, breaks = x_breaks) +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(
        title = paste("Threshold vs Accuracy", title_suffix, "—", tools::toTitleCase(input$method)),
        subtitle = input$experiment, x = "Threshold", y = "Accuracy"
      ) +
      theme_classic(base_size = 14) +
      theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(hjust = 0.5))
  }
  
  output$scorePlot <- renderPlot({ make_single_plot(acc_score(), "(score only)") })
  output$fullPlot  <- renderPlot({ make_single_plot(acc_full(),  "(score + match)") })
  
  render_confusion_ui <- function(method, threshold) {
    df <- selected_data()
    cm <- compute_confusion_matrix(df, method, threshold)
    tagList(
      tags$h4(paste(tools::toTitleCase(method), "Confusion Matrix")),
      tags$p(conf_subtitle(df, method, threshold), style = "font-size: 90%; color: gray;"),
      tableOutput(paste0("conf_", method, "_table"))
    )
  }
  
  output$conf_fuzzy <- renderUI({ render_confusion_ui("fuzzy", input$fuzzy_thresh) })
  output$conf_tfidf <- renderUI({ render_confusion_ui("tfidf", input$tfidf_thresh) })
  output$conf_embed <- renderUI({ render_confusion_ui("embed", input$embed_thresh) })
  
  output$conf_fuzzy_table <- renderTable({ compute_confusion_matrix(selected_data(), "fuzzy", input$fuzzy_thresh) }, rownames = TRUE)
  output$conf_tfidf_table <- renderTable({ compute_confusion_matrix(selected_data(), "tfidf", input$tfidf_thresh) }, rownames = TRUE)
  output$conf_embed_table <- renderTable({ compute_confusion_matrix(selected_data(), "embed", input$embed_thresh) }, rownames = TRUE)
}

shinyApp(ui, server)
